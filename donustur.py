import os
import json
from PIL import Image

def yolo_to_coco(yolo_dir, images_dir, output_dir, categories):
    """
    YOLO formatından COCO formatına dönüştürme.
    
    Parameters:
    - yolo_dir: YOLO etiket dosyalarının bulunduğu klasör
    - images_dir: Görsellerin bulunduğu klasör
    - output_dir: Çıktının kaydedileceği klasör
    - categories: Sınıf id ve adlarının listesi [{'id': 1, 'name': 'cat'}, ...]
    """
    # Çıktı klasörünü oluştur
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    annotations = {"images": [], "annotations": [], "categories": categories}
    annotation_id = 1

    # YOLO etiket dosyalarını oku
    for idx, txt_file in enumerate(os.listdir(yolo_dir)):
        if not txt_file.endswith(".txt"):
            continue

        # Görsel dosyasını bul
        base_name = txt_file.replace(".txt", "")
        image_path = None

        for ext in [".jpg", ".webp", ".png", ".jpeg"]:
            potential_path = os.path.join(images_dir, base_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if not image_path:
            print(f"Görsel bulunamadı: {base_name}")
            continue

        # Görselin boyutlarını al
        with Image.open(image_path) as img:
            width, height = img.size

        # Görsel bilgilerini ekle
        image_id = idx + 1
        annotations["images"].append({
            "id": image_id,
            "file_name": f"{base_name}.jpg",  # Çıkış görseli .jpg olacak
            "width": width,
            "height": height
        })

        # Görseli çıktı klasörüne kopyala ve .jpg formatına çevir
        output_image_path = os.path.join(output_dir, "train", f"{base_name}.jpg")
        with Image.open(image_path) as img:
            img.convert("RGB").save(output_image_path, "JPEG")

        # Etiket dosyasını oku
        with open(os.path.join(yolo_dir, txt_file), "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)
            x_min = (x_center - bbox_width / 2) * width
            y_min = (y_center - bbox_height / 2) * height
            bbox_width = bbox_width * width
            bbox_height = bbox_height * height

            # Annotation bilgilerini ekle
            annotations["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(class_id),
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            })
            annotation_id += 1

    # JSON dosyasını kaydet
    with open(os.path.join(output_dir, "annotations_train.json"), "w") as f:
        json.dump(annotations, f, indent=4)
    print(f"COCO formatı başarıyla {output_dir} klasörüne kaydedildi!")

# Kullanım
yolo_to_coco(
    yolo_dir="labels\\train",  # YOLO etiket klasörü
    images_dir="images\\train",  # Görsellerin olduğu klasör
    output_dir="faster_rcnn_modeli",  # Çıkış klasörü
    categories=[
        {"id": 1, "name": "sinif1"},
        {"id": 2, "name": "sinif2"},
        # Kendi sınıflarınızı buraya ekleyin
    ]
)
