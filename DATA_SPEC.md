# DATA_SPEC

> Fill this from Huawei's example notebook and the actual local dataset.  
> Do not guess conventions.

## Dataset root
- Local (not in Git): C:\Users\ChPra\summer 26\huawei_data_complete\2026 Munich Tech Arena - Datas

## Directory layout
- mesh/  (subject meshes, .ply)
- landmarks/  (ear landmarks, .csv)

## Subject ID convention
- Format: Pxxxx (example: P0001)
- IDs are non-contiguous (some numbers skipped)

## Mesh filename convention
- <SUBJECT_ID>.ply
- Example: P0001.ply

## Annotation filename convention
- <SUBJECT_ID>_left_ear_landmarks.csv
- <SUBJECT_ID>_right_ear_landmarks.csv

## PLY fields
- Vertices:
- Faces:
- Vertex normals:
- Colours/other:

## Landmark format
- Shape: (85, 3) per ear file
- Dtype: float (parsed to float64 in current scripts)
- Units:
- Left file/key: <SUBJECT_ID>_left_ear_landmarks.csv
- Right file/key: <SUBJECT_ID>_right_ear_landmarks.csv

## Verified contour indices
- Outer helix:
- Concha outline:
- Inner helix:
- Superior antihelix:

## Coordinate system
- X: back of head -> front (nose direction)
- Y: left ear canal -> right ear canal
- Z: upward
- Approximate ranges: X [-40.39, 14.29], Y [-105.96, 107.76], Z [-38.01, 44.68]

## Left/right verification
- Verified file pairing exists for usable subjects (mesh + left + right).

## Mirror convention
- Is one side mirrored?
- Axis/sign:
- Visual evidence:

## Frozen train/validation split
- Seed: 42
- Train subject count: 160
- Val subject count: 40
- File storing IDs: configs/split_seed42.json

## Crop configuration
- Left bounds:
- Right bounds:
- Margins:
- Training statistics used:
- Sanity-check thresholds:

## Canonical transform
- Centre definition:
- Scale definition:
- Mirror rule:
- Transform formula:
- Inverse formula:

## Point sampling
- N:
- Method:
- Seed behaviour:

## Normals
- Source:
- Normalisation:

## Output format required by challenge
- {"left": np.ndarray((85,3)), "right": np.ndarray((85,3))}
- Coordinates in original Huawei global frame

## Known anomalies / edge cases
- Subject IDs are non-contiguous

## Last updated
07/09/2026
