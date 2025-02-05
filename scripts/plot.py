import matplotlib.pyplot as plt
import numpy as np
'''
# 3D U-Net
epochs = np.arange(1, 51, 2)  # 51 epochs
training_loss = [
  0.3433, 0.3111, 0.3041, 0.3030, 0.2941, 0.2953, 0.2821, 0.2812, 0.2781, 0.2759, 0.2738, 0.2645, 0.2614, 0.2540,
    0.2480, 0.2363, 0.2347, 0.2431, 0.2299, 0.2177, 0.2116, 0.2070, 0.2028, 0.1929, 0.2124
  ]
training_loss = [x * 100 for x in training_loss]

validation_loss = [
  0.3054, 0.2918, 0.3064, 0.2855, 0.2887, 0.2989, 0.3044, 0.2682, 0.2692, 0.2820, 0.2651, 0.2599, 0.2420, 0.2348,
    0.2216, 0.2292, 0.2287, 0.2308, 0.2217, 0.3024, 0.2324, 0.2049, 0.1892, 0.2016,  0.1881
  ]
validation_loss = [x * 100 for x in validation_loss]


iou_bone = [
  0.2592, 0.2301, 0.2093, 0.2312, 0.2048, 0.2312, 0.3556, 0.2491, 0.2681, 0.3294, 0.2502, 0.3587, 0.2686, 0.3164,
    0.3222, 0.2557, 0.3332, 0.3655, 0.2894, 0.2126, 0.2557, 0.2887, 0.3573, 0.3429, 0.3648
  ]
iou_bone = [x * 100 for x in iou_bone]


dice_bone = [
  0.4116, 0.3741, 0.3462, 0.3756,  0.3400, 0.3756, 0.5246, 0.3988, 0.4229, 0.4956, 0.4003, 0.5280, 0.4235, 0.4807,
    0.4874, 0.4073, 0.4998, 0.5353, 0.4489, 0.3507, 0.4073, 0.4481, 0.5265, 0.5107, 0.5346
]
dice_bone = [x * 100 for x in dice_bone]


patches = [64,80,64,64,64,80,80,80,80,80,80,80,112,112,32,32,144,16,16,176,176,80,80,128,128,80,80,80,80,80,80,96,96
]'''

# Attention 3D U-Net
epochs = np.arange(1, 41, 2)  # 51 epochs
training_loss = [
  0.4845, 0.4180, 0.4086, 0.3925, 0.3801, 0.3625, 0.3417, 0.3183, 0.3128, 0.3072, 0.3039, 0.2925, 0.2894, 0.2862,
  0.2727, 0.2733, 0.2635, 0.2581, 0.2756, 0.2566
  ]
training_loss = [x * 100 for x in training_loss]

validation_loss = [
  0.4307, 0.4712, 0.3978, 0.4485, 0.5888, 0.3695,  0.4658, 0.6657, 0.4731, 0.3659, 0.6337, 0.3925, 0.3181, 0.4579,
  0.4866, 0.3927, 0.8481, 0.4721, 0.4712, 0.3406
  ]
validation_loss = [x * 100 for x in validation_loss]

iou_bone = [
  0.3270, 0.2055, 0.3679, 0.2459, 0.1517, 0.4475, 0.2131, 0.1296, 0.1790, 0.4015, 0.1330, 0.2444, 0.3513, 0.1641,
  0.1468, 0.2160, 0.1315, 0.1760, 0.4628, 0.5060
  ]
iou_bone = [x * 100 for x in iou_bone]

dice_bone = [
  0.4928, 0.3409, 0.5379, 0.3948, 0.2635, 0.6183, 0.3514, 0.2295, 0.3036, 0.5730, 0.2348, 0.3928, 0.5199, 0.2819,
  0.2561, 0.3552, 0.2324, 0.2994, 0.6328, 0.6719
]
dice_bone = [x * 100 for x in dice_bone]


# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, training_loss, label="Training Loss", linestyle="--")
plt.plot(epochs, validation_loss, label="Validation Loss", color="blue", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Loss (%)")
plt.ylim(0, 100)
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Plot IoU and Dice Scores
plt.figure(figsize=(10, 5))

plt.plot(epochs, iou_bone, label="IoU", color="purple",linestyle="-")
plt.plot(epochs, dice_bone, label="Dice Score", color="green",linestyle="-")

plt.xlabel("Epochs")
plt.ylabel("IoU/Dice Score (%)")
plt.ylim(0, 100)
plt.title("IoU and Dice Scores Over Epochs Training and Validation")
plt.legend()
plt.grid(True)
plt.show()



'''
# 9 classes
epochs = np.arange(1, 41, 2)  # 51 epochs
training_loss = [
  62.74, 85, 86, 82, 83, 81, 82, 81, 81, 81.8, 81, 79, 77, 79, 78, 76, 74, 76, 75, 72
  ]


validation_loss = [
    84, 81, 83, 80, 84, 80, 79, 82, 78.9, 83, 82, 77.6, 79, 76.5, 76.6, 77, 74, 75, 74, 73
  ]

iou_c1 = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0, 0.32, 0.01, 1.06, 1.23, 0.01, 1.06, 1.90, 0.2, 1.67
]

dice_c1 = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0, 0.64, 0.01, 2.10, 2.43, 2.44, 2.10, 3.73, 0.4, 3.29
]

iou_c2 = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0.03, 0.41, 0.46, 0.03, 1.35, 3.15, 0.19, 1.14
]

dice_c2 = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03, 0.01, 0.01, 0.07, 0.82, 0.92, 3.63, 2.66, 6.10, 0.38, 2.26
]

iou_c3 = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

dice_c3 = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

iou_c4 = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0.02, 0.05, 0, 0, 0, 0, 0
]

dice_c4 = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0.03, 0.11, 0, 0, 0, 0, 0
]

iou_c5 = [
  0, 0, 0.18, 0.20, 0, 0, 0.12, 0.49, 0.2, 0.63, 0.58, 0.76, 2.77, 2.77, 2.08, 1.78, 3.41, 3.89, 1.21, 4.26
]

dice_c5 = [
  0, 0, 0.36, 0.02, 0, 0, 0.24, 0.49, 0.4, 1.26, 1.15, 1.50, 5.38, 5.40, 4.07, 3.5, 6.6, 7.49, 2.40, 8.18

]

iou_c6 = [
  0, 0, 0, 0, 4.76, 5.60, 5.67, 1.60, 4.49, 3.47, 4.15, 4.23, 2.73, 6.20, 7.91, 4.58, 6.09, 7.77, 3.20, 4.72
]

dice_c6 = [
  0, 0, 0, 0, 9.09, 10.61, 10.73, 3.14, 8.60, 6.71, 7.97, 8.12, 5.31, 11.68, 14.65, 8.76, 11.49, 14.42, 6.21, 9.01
]

iou_c7 = [
  0, 3.97, 2.09, 5.09, 1.62, 3.39, 3.45, 2.66, 2.68, 1.65, 2.74, 2.16, 5.62, 4.28, 4.63, 4.11, 4.17, 4.21, 3.04, 3.90
]

dice_c7 = [
  0, 7.63, 4.10, 9.69, 3.18, 6.56, 6.68, 5.18, 5.22, 3.25, 5.34, 4.23, 10.63, 8.20, 8.85, 7.89, 8, 8.09, 5.91, 7.50
]

iou_c8 = [
  0, 0, 0.08, 1.28, 0.05, 2.29, 0, 3.55, 3.25, 2.96, 4.06, 5.65, 4.76, 5.51, 5.96, 7.13, 5.28, 4.25, 4.24, 7.02
]

dice_c8 = [
  0, 0, 1.60, 2.52, 0.01, 4.47, 0, 6.85, 6.29, 5.70, 7.80, 10.69, 9.08, 0.45, 11.25, 13.31, 10.04, 8.16, 8.14, 13.11
]



# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, training_loss, label="Training Loss", color="blue", linestyle="--")
plt.plot(epochs, validation_loss, label="Validation Loss", color="blue", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Loss (%)")
plt.ylim(0, 100)
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Plot IoU and Dice Scores
plt.figure(figsize=(10, 5))
plt.plot(epochs, iou_c1, label="Hand L IoU",linestyle="-")
plt.plot(epochs, dice_c1, label="Hand L Dice",linestyle="--")
plt.plot(epochs, iou_c2, label="Hand R IoU",linestyle="-")
plt.plot(epochs, dice_c2, label="Hand R Dice",linestyle="--")
plt.plot(epochs, iou_c3, label="Humerus L IoU",linestyle="-")
plt.plot(epochs, dice_c3, label="Humerus L Dice",linestyle="--")
plt.plot(epochs, iou_c4, label="Humerus R IoU",linestyle="-")
plt.plot(epochs, dice_c4, label="Humerus R Dice",linestyle="--")
plt.plot(epochs, iou_c5, label="Radius L IoU",linestyle="-")
plt.plot(epochs, dice_c5, label="Radius L Dice",linestyle="--")
plt.plot(epochs, iou_c6, label="Radius R IoU",linestyle="-")
plt.plot(epochs, dice_c6, label="Radius R Dice",linestyle="--")
plt.plot(epochs, iou_c7, label="Ulna L IoU",linestyle="-")
plt.plot(epochs, dice_c7, label="Ulna L Dice",linestyle="--")
plt.plot(epochs, iou_c8, label="Ulna R IoU",linestyle="-")
plt.plot(epochs, dice_c8, label="Ulna R Dice",linestyle="--")



plt.xlabel("Epochs")
plt.ylabel("IoU/Dice Score (%)")
plt.ylim(0, 100)
plt.title("IoU and Dice Scores Over Epochs Training and Validation")
plt.legend()
plt.grid(True)
plt.show()
'''