import gradio as gr
import numpy as np
import cv2
from PIL import Image
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ─── initialize your SAM2 predictor ───────────────────────────────────────────
checkpoint = "/home/zhengxiao-han/projects/foundation_models/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg   = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor   = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# ─── helpers & callbacks ──────────────────────────────────────────────────────

def get_point_radius(img: np.ndarray, pct: float = 0.01, min_px: int = 2) -> int:
    h, w = img.shape[:2]
    return max(min_px, int(min(h, w) * pct))

def load_and_setup(img_np):
    predictor.set_image(img_np)
    return img_np, [], [], img_np, None

def add_point(evt: gr.SelectData, mode, orig_img, fg_pts, bg_pts):
    x, y = int(evt.index[0]), int(evt.index[1])
    if mode == "Foreground":
        fg_pts.append([x, y])
    else:
        bg_pts.append([x, y])

    # Run inference
    predictor.set_image(orig_img)
    pts  = np.array(fg_pts + bg_pts)
    labs = np.array([1] * len(fg_pts) + [0] * len(bg_pts))
    masks, _, _ = predictor.predict(
        point_coords=pts.reshape(-1, 2),
        point_labels=labs,
        multimask_output=True,
        return_logits=False,
    )

    # Create a single binary mask (any pixel in any predicted mask)
    combined = np.any(masks, axis=0).astype(np.uint8)
    # Overlay that combined mask exactly once (no varying depths)
    overlay = orig_img.copy()
    cm = (combined[:, :, None] * np.array([0, 0, 255])).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 1.0, cm, 0.35, 0)

    # Draw points **on top** of the mask overlay
    r = get_point_radius(overlay)
    for px, py in fg_pts:
        cv2.circle(overlay, (px, py), r, (0, 0, 255), -1)
    for px, py in bg_pts:
        cv2.circle(overlay, (px, py), r, (0, 255, 0), -1)

    return overlay, fg_pts, bg_pts, orig_img, masks

def clear_last(mode, orig_img, fg_pts, bg_pts):
    if mode == "Foreground" and fg_pts:
        fg_pts.pop()
    elif mode == "Background" and bg_pts:
        bg_pts.pop()

    # Re-run inference
    predictor.set_image(orig_img)
    pts  = np.array(fg_pts + bg_pts)
    labs = np.array([1] * len(fg_pts) + [0] * len(bg_pts))
    masks, _, _ = predictor.predict(
        point_coords=pts.reshape(-1, 2),
        point_labels=labs,
        multimask_output=True,
        return_logits=False,
    )

    # Overlay masks
    overlay = orig_img.copy()
    for m in masks:
        cm = (m[:, :, None] * np.array([0, 0, 255])).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 1.0, cm, 0.35, 0)

    # Draw points **on top** of the mask overlay
    r = get_point_radius(overlay)
    for px, py in fg_pts:
        cv2.circle(overlay, (px, py), r, (0, 0, 255), -1)
    for px, py in bg_pts:
        cv2.circle(overlay, (px, py), r, (0, 255, 0), -1)

    return overlay, fg_pts, bg_pts, orig_img, masks

def export_mask(masks, filepath):
    # OR all masks together so it covers every predicted region
    combined = np.any(masks, axis=0).astype(np.uint8) * 255
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(combined).save(filepath)
    return f"✅ Saved mask to `{filepath}`"


# ─── build the Gradio app ────────────────────────────────────────────────────

with gr.Blocks() as demo:
    gr.Markdown("## SAM2 Interactive Segmentation")

    with gr.Row():
        # LEFT: image
        with gr.Column(scale=2):
            img_in = gr.Image(
                label="Upload Image",
                sources=["upload"],
                type="numpy",
                interactive=True,
                width=1024,
                height=768,
            )
        # RIGHT: controls
        with gr.Column(scale=1):
            mode       = gr.Radio(["Foreground", "Background"], label="Click Mode", value="Foreground")
            clear_btn  = gr.Button("Clear Last Point")
            filepath   = gr.Textbox(label="Export Path & Filename", value="./mask.png", placeholder="e.g. /home/user/output/my_mask.png")
            export_btn = gr.Button("Export Mask")
            status     = gr.Textbox(label="Export Status")

    # hidden states
    fg_state   = gr.State([])
    bg_state   = gr.State([])
    original   = gr.State(None)
    mask_state = gr.State(None)

    img_in.upload(
        fn=load_and_setup,
        inputs=[img_in],
        outputs=[img_in, fg_state, bg_state, original, mask_state],
    )
    img_in.select(
        fn=add_point,
        inputs=[mode, original, fg_state, bg_state],
        outputs=[img_in, fg_state, bg_state, original, mask_state],
    )
    clear_btn.click(
        fn=clear_last,
        inputs=[mode, original, fg_state, bg_state],
        outputs=[img_in, fg_state, bg_state, original, mask_state],
    )
    export_btn.click(
        fn=export_mask,
        inputs=[mask_state, filepath],
        outputs=[status],
    )

demo.launch()
