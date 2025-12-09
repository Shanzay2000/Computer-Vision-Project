import xml.etree.ElementTree as ET
import numpy as np
import json
import os

#agisoft gave .xml so we changed it to all_cameras.json as it was required for web
XML_FILE = "cameras_metashape.xml"
OUTPUT_JSON = "all_cameras.json"

def parse_metashape_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    cameras = []

    for cam in root.findall(".//camera"):
        label = cam.get("label") or cam.get("id")

        transform_el = cam.find("transform")
        if transform_el is None:
            continue

        vals = list(map(float, transform_el.text.split()))
        M = np.array(vals, dtype=float).reshape(4, 4)

        R = M[:3, :3]
        t = M[:3, 3]

        cameras.append({
            "label": label,
            "R": R,
            "t": t
        })

    return cameras

def build_json(cameras):
    out = []

    for cam in cameras:
        image_name = os.path.basename(cam["label"])

        out.append({
            "rotation": cam["R"].tolist(),
            "translation": cam["t"].tolist(),
            "image": image_name
        })

    return {"cameras": out}

def main():
    cams = parse_metashape_xml(XML_FILE)
    data = build_json(cams)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data['cameras'])} cameras to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
