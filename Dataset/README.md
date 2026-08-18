# NEU Metal Surface Defects Dataset - Week 1

## Dataset Information
- **Source**: NEU Metal Surface Defects Database
- **Date Organized**: August 15, 2026
- **Organized by**: Chaitanya (Member 1)
- **Location**: Stored locally (not in GitHub)

## Actual Dataset Statistics
- **Total Images**: 1800
- **Training Images**: 1440 (80%)
- **Validation Images**: 360 (20%)
- **Image Format**: .jpg, .png
- **Annotation Format**: .txt (YOLO format with normalized bounding boxes)

## Local Folder Structure (Chaitanya's Computer)

Local path: `C:\Users\revat\Downloads\Neu zaalima dataset\`


## Defect Classes (6 types)
1. **Crazing** - Surface cracks
2. **Inclusion** - Foreign material embedded
3. **Patches** - Discolored areas
4. **Pitted Surface** - Small holes/depressions
5. **Rolled-in Scale** - Oxide scale
6. **Scratches** - Linear marks

## Data Verification Checklist ✅
- [x] Downloaded from NEU Metal Surface Defects Database
- [x] Total image count verified: 1800 images
- [x] Training/Validation split confirmed: 80/20
- [x] Image-annotation matching verified: All images have corresponding annotations
- [x] All 6 defect classes present in dataset
- [x] YOLO format annotations verified
- [x] No missing or corrupted files detected

## Dataset Split Details
- **Training Set**: 1440 images (80%) + 1440 annotations
- **Validation Set**: 360 images (20%) + 360 annotations
- **Split Ratio**: 80% training, 20% validation
- **Total Annotations**: 1800 (one annotation per image - verified)

## File Matching Verification
✅ **VERIFIED** - Every image has a matching annotation file
- Training set: 1440/1440 images matched with labels (100%)
- Validation set: 360/360 images matched with labels (100%)
- No orphan images or labels found

## Storage Notes
⚠️ **Large image files stored locally, NOT in GitHub**
- Team members keep their own local copy
- Only documentation is uploaded to GitHub
- Dataset stored locally at: `C:\Users\revat\Downloads\Neu zaalima dataset\NEU-DET\`

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
✅ All images matched with annotations (verified)
✅ Image-label verification complete
✅ Ready for Week 2 training phase

## Next Steps
- Week 2: YOLOv8 training setup and model training
- Member 2 (Arun): Complete annotation validation report
- Member 3 (Musthafa): Finalize augmentation pipeline
- Member 4 (Nallapa): Complete mAP evaluation documentation