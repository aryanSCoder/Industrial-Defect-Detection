# NEU Metal Surface Defects Dataset - Week 1

## Dataset Information
- **Source**: NEU Metal Surface Defects Database
- **Date Organized**: August 15, 2026
- **Organized by**: Chaitanya (Member 1)
- **Storage**: Local (not in GitHub - large image files)

## Dataset Statistics
- **Total Images**: 1800
- **Training Images**: 1440 (80%)
- **Validation Images**: 360 (20%)
- **Image Format**: .jpg, .png
- **Annotation Format**: .txt (YOLO format with normalized bounding boxes)

## Local Folder Structure (Chaitanya's Computer)

**Local path**: `C:\Users\revat\Downloads\Neu zaalima dataset\NEU-DET\`


## Defect Classes (6 types)
1. **Crazing** - Surface cracks
2. **Inclusion** - Foreign material embedded
3. **Patches** - Discolored areas
4. **Pitted Surface** - Small holes/depressions
5. **Rolled-in Scale** - Oxide scale
6. **Scratches** - Linear marks

## Verification Checklist ✅
- [x] Downloaded from NEU Metal Surface Defects Database
- [x] Total image count verified: 1800 images
- [x] Training/Validation split confirmed: 1440/360 (80/20)
- [x] Image-annotation matching verified: All images have matching annotation files
- [x] All 6 defect classes present in dataset
- [x] YOLO format annotations verified
- [x] No missing or corrupted files detected

## Image-Label Verification Details
- **Training set**: 1440/1440 images matched with labels (100%)
- **Validation set**: 360/360 images matched with labels (100%)
- **Total annotations**: 1800 (one annotation per image - verified)
- **Orphan files**: None found

## YOLO Configuration
```yaml
path: C:\Users\revat\Downloads\Neu zaalima dataset\NEU-DET
train: train/images
val: validation/images

nc: 6
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']
```

## Status
✅ Dataset downloaded and organized
✅ Split into train/validation (80/20)
✅ All 1800 images matched with annotations (verified)
✅ Ready for Week 2 training phase

## Team Workflow
- **Member 1 (Chaitanya)**: Dataset organization ✅
- **Member 2 (Arun)**: Annotation validation
- **Member 3 (Musthafa)**: Data augmentation
- **Member 4 (Nallapa)**: mAP evaluation documentation
- **Lead (Aryan)**: Integration and review