# Mock vs Real on `bus.jpg`

Same Ultralytics sample `data/gold/images_real/bus.jpg` (people + bus).

| | Mock | Real |
|--|------|------|
| run_id | see `meta.json` | see `meta.json` |
| experts | `ExpertA_mock_sam3`, `ExpertB_mock_dino_ovd` | `ExpertA_yolo_world` (`yolov8s-worldv2.pt`), `ExpertB_owlv2` (`google/owlv2-base-patch16`) |
| boxes | Deterministic fake boxes in relative image regions — **do not** land on the bus/people | YOLO-World / OWLv2 boxes cover the bus body and standing people (xyxy in original pixels) |
| `raw_output.mock` | `true` | `false` |
| viz | `mock/bus_boxes.png` | `real/bus_boxes.png` |

**Takeaway:** mock is only for pipeline regression on synthetic gold; open-vocab quality needs real images + real experts. These real experts are **paper substitutes** (not SAM3 / in-house DINO).
