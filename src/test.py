import os
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
import mmcv


def main():
    # === 設定 ===
    config_path = "configs/detr/detr_r50_8xb2-150e_coco.py"  # mmdetection内のパス
    checkpoint_path = "checkpoint/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth"
    image_path = "data/sample/R0012741.JPG"  # 推論対象の画像ファイル
    output_path = "output/test.jpg"

    # === モデルの初期化 ===
    model = init_detector(config_path, checkpoint_path, device="cuda:0")

    # === 推論 ===
    result = inference_detector(model, image_path)

    # === 結果の描画と保存 ===
    img = mmcv.imread(image_path, channel_order="rgb")

    # Visualizerの準備
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = model.dataset_meta  # クラス情報などをセット

    # 描画＆保存
    visualizer.add_datasample(
        name="dab_detr_result",
        image=img,
        data_sample=result,
        draw_gt=False,
        show=False,
        wait_time=0,
        out_file=output_path,
        pred_score_thr=0.8,
    )

    print(f"✅ 推論完了。出力画像: {output_path}")


if __name__ == "__main__":
    main()
