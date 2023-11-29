# Migration-Index-Calculation-for-Celebral-Palsy-Patients-From-Pelvis-X-Ray
![Migration Index](https://github.com/tirthvyas7/Migration-Index-Calculation-for-Celebral-Palsy-Patients-From-Pelvis-X-Ray/assets/132454970/bdce5242-31ff-4360-a854-5758aaa76ac0)<br>
> Image Credits: https://www.orthobullets.com/pediatrics/4130/cerebral-palsy--hip-conditions
## Introduction
This project aims to find Migration Index exact value calculation for Cerebral Palsy Patients using some inputs from Doctor.This code was implemented in python using OpenCV library.
The Computer Vision techniques used in this code are as follows:
### For Initial Segmentation of Bones:
- Thresholding and Bitwise AND Operator
### For Calculation of 'A' Value:
- Harris Corner Detector
### For Calculation of 'B' Value:
- Contour Detection
- Maximum Contour bounding box
## Steps to use this Project for any Doctor:
1. The first popup will show the input X-Ray.
2. On pressing ENTER one more popup will show which will ask the doctor to crop Acetabulum and Femoral Head in one frame of whichever hip he wants for calculation of **A**.
3. One more popup will show the possible corners for the calculation of **A**, doctor has to choose the most preferable corner from outer side of Femoral Head and Acetabulum by single clicking on that corner point. **A** value will be calculated as per selection of doctor.
4. The next step is to crop Femoural Head of doctor's choice from the Image for **B** value calculation. On doing this, the maximum area bounding box will be made around Femoral Head and **B** value will be calculated.
5. At the end the Caolculated Migration index can be seen on the terminal window.


