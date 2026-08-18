# NEU Metal Surface Defects Dataset - Week 1

## Overview
- **Source**: NEU Metal Surface Defects Database
- **Organized by**: Chaitanya (Member 1)
- **Date**: August 15, 2026
- **Storage**: Local only (images not in GitHub)

## Dataset Statistics
- **Total**: 1800 images
- **Training**: 1440 images (80%) + 1440 annotations
- **Validation**: 360 images (20%) + 360 annotations
- **Format**: YOLO (.txt files, normalized coordinates)

## Local Folder Structure (Chaitanya's Computer)

Local path: `C:\Users\revat\Downloads\Neu zaalima dataset\NEU-DET\`


## Defect Classes (6 types)
1. Crazing - Surface cracks
2. Inclusion - Foreign material
3. Patches - Discolored areas
4. Pitted Surface - Small holes
5. Rolled-in Scale - Oxide scale
6. Scratches - Linear marks

## Verification ✅
- ✅ Dataset downloaded from NEU
- ✅ All 1800 images verified
- ✅ Image-annotation matching: 100% (1440 train, 360 validation)
- ✅ All 6 defect classes present
- ✅ YOLO format verified
- ✅ No corrupted files

## YOLO Configuration
```yaml
path: C:\Users\revat\Downloads\Neu zaalima dataset\NEU-DET
train: train/images
val: validation/images
nc: 6
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']
```

## Status
✅ Complete and ready for Week 2 training