import numpy as np
import matplotlib.pyplot as plt

# for RPI2241 and NPInter v2.0
categories = ['ACC', 'SEN', 'SPE', 'PRE', 'F1', 'MCC', 'AUC']
values_rpi2241 = [
    [77.4, 75.0, 79.7, 79.5, 76.8, 55.2, 84.9],
    [73.7, 82.1, 72.4, 79.7, 76.5, 68.9, 90.4],
    [82.3, 83.3, 79.2, 80.3, 82.9, 65.1, 89.9],
    [87.6, 84.7, 89.6, 89.4, 87.6, 75.5, 93.5],
    [76.7, 72.7, 80.8, 80.7, 69.7, 63.8, 84.6],
    [76.2, 71.1, 81.3, 80.4, 75.5, 62.8, 85.8],
    [87.6, 81.7, 90.9, 90.4, 86.9, 75.9, 93.3],
    [88.2, 85.1, 93.6, 92.8, 87.6, 76.1, 93.8],
    [94.4, 92.1, 96.8, 96.6, 94.3, 89.0, 97.2],
]
values_npinter = [
    [83.6, 85.8, 81.5, 82.2, 84.0, 67.4, 91.6],
    [83.7, 85.1, 82.4, 79.7, 86.5, 78.9, 82.8],
    [89.5, 85.7, 91.4, 92.2, 87.2, 81.4, 93.9],
    [91.3, 89.8, 94.2, 94.9, 90.8, 83.3, 96.8],
    [81.4, 79.7, 85.4, 87.6, 84.3, 72.9, 82.4],
    [83.4, 80.3, 83.2, 82.9, 80.7, 73.3, 86.5],
    [91.6, 88.9, 94.3, 94.1, 91.4, 83.5, 97.3],
    [91.9, 89.5, 95.7, 95.2, 91.7, 83.9, 97.6],
    [95.4, 94.9, 95.8, 95.8, 95.4, 90.8, 98.5],
]
feature_names = [
    "LPI-CSFFR",
    "LPI-CNNCP",
    "Capslule-LPI",
    "EDLMFC",
    "CFRP",
    "IPMiner",
    "RPITER",
    "RPI-EDLCN",
    "Transfer-RPI",
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={"projection": "polar"})

# Calculate angles
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # Complete the loop


colors = ["#6699FF", "#98FB98", "#FFD580", "#FF6666", "#ADFF2F", "#00BFFF", "#FFFF00", "#FFA500", "#FF0000"]


# Adjusted bar width for no gaps
bar_width = (2 * np.pi) / (len(categories) * len(values_rpi2241) + len(values_rpi2241))

# Plot bars for RPI2241 dataset on the left subplot (ax1)
for idx, (feature_data, color) in enumerate(zip(values_rpi2241, colors)):
    data = feature_data + feature_data[:1]  # Complete the loop
    adjusted_angles = [
        angle - (bar_width * (len(values_rpi2241) - 1)) / 2 + idx * bar_width
        for angle in angles[:-1]
    ] + [
        angles[0] - (bar_width * (len(values_rpi2241) - 1)) / 2 + idx * bar_width
    ]
    ax1.bar(
        adjusted_angles,
        data,
        color=color,
        width=bar_width,
        label=feature_names[idx],
        alpha=0.85,
        zorder=1,
    )
    for angle, value in zip(adjusted_angles, feature_data):
        angle_deg = (np.degrees(np.pi / 2 - angle)) % 360
        if 90 < angle_deg < 270:
            # Left side
            rotation = angle_deg + 180  # Rotate text to align with the radius
            r = value - 12  # Slightly inside the bar
            alignment = 'right'
        else:
            # Right side
            rotation = angle_deg
            r = value - 12  # Slightly outside the bar
            alignment = 'left'
        ax1.text(
            angle,
            r,
            f"{value:.1f}",
            color="black",
            size=14,
            ha=alignment,
            va="center",
            rotation=rotation,
            rotation_mode='anchor',
            zorder=4,
        )

ax1.xaxis.grid(True, color="gray", linestyle="-", linewidth=1.0, zorder=1)  # Solid line
ax1.yaxis.grid(True, color="gray", linestyle="--", linewidth=0.5, zorder=1)  # Dashed line
ax1.yaxis.set_tick_params(labelsize=14)

# Beautify the first plot (ax1)
ax1.set_theta_offset(np.pi / 2)
ax1.set_theta_direction(-1)
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(categories, fontsize=16)
ax1.set_title("RPI2241", fontsize=20)

# Plot bars for NPInter v2.0 dataset on the right subplot (ax2)
for idx, (feature_data, color) in enumerate(zip(values_npinter, colors)):
    data = feature_data + feature_data[:1]  # Complete the loop
    adjusted_angles = [
        angle - (bar_width * (len(values_npinter) - 1)) / 2 + idx * bar_width
        for angle in angles[:-1]
    ] + [
        angles[0] - (bar_width * (len(values_npinter) - 1)) / 2 + idx * bar_width
    ]
    ax2.bar(
        adjusted_angles,
        data,
        color=color,
        width=bar_width,
        label=feature_names[idx],
        alpha=0.85,
        zorder=1,
    )
    for angle, value in zip(adjusted_angles, feature_data):
        angle_deg = (np.degrees(np.pi / 2 - angle)) % 360
        if 90 < angle_deg < 270:
            # Left side
            rotation = angle_deg + 180  # Rotate text to align with the radius
            r = value - 12  # Slightly inside the bar
            alignment = 'right'
        else:
            # Right side
            rotation = angle_deg
            r = value - 12  # Slightly outside the bar
            alignment = 'left'
        ax2.text(
            angle,
            r,
            f"{value:.1f}",
            color="black",
            size=14,
            ha=alignment,
            va="center",
            rotation=rotation,
            rotation_mode='anchor',
            zorder=4,
        )

ax2.xaxis.grid(True, color="gray", linestyle="-", linewidth=1.0, zorder=1)  # Solid line
ax2.yaxis.grid(True, color="gray", linestyle="--", linewidth=0.5, zorder=1)  # Dashed line
ax2.yaxis.set_tick_params(labelsize=14)


ax2.set_theta_offset(np.pi / 2)
ax2.set_theta_direction(-1)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, fontsize=16)
ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1), fontsize=16, frameon=False)
ax2.set_title("NPInter v2.0", fontsize=20)

plt.tight_layout()
plt.show()




# import numpy as np
# import matplotlib.pyplot as plt
#
# # Data for RPI1807
# data_rpi1807 = [
#     [90.7, 93.9, 81.5, 93.2, 93.8, 75.5, 95.3],
#     [92.3, 95.2, 81.7, 93.5, 93.8, 79.6, 94.8],
#     [92.8, 95.5, 82.7, 94.3, 93.9, 81.4, 95.2],
#     [93.7, 96.8, 84.6, 94.8, 95.8, 83.3, 96.3],
#     [93.0, 97.6, 78.7, 93.1, 95.8, 80.3, 96.4],
#     [93.5, 98.3, 77.8, 93.1, 94.7, 82.8, 89.3],
#     [93.5, 97.1, 82.8, 94.5, 95.6, 83.2, 96.8],
#     [93.8, 97.4, 84.2, 94.8, 95.9, 83.3, 97.4],
#     [94.3, 96.8, 86.9, 95.6, 96.2, 84.8, 98.1],
# ]
#
# categories = ['ACC', 'SEN', 'SPE', 'PRE', 'F1', 'MCC', 'AUC']
# feature_names = [
#     "LPI-CSFFR",
#     "LPI-CNNCP",
#     "Capslule-LPI",
#     "EDLMFC",
#     "CFRP",
#     "IPMiner",
#     "RPITER",
#     "RPI-EDLCN",
#     "Transfer-RPI",
# ]
#
# fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": "polar"})
#
# angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
# angles += angles[:1]  # Complete the loop
#
# colors = ["#6699FF", "#98FB98", "#FFD580", "#FF6666", "#ADFF2F", "#00BFFF", "#FFFF00", "#FFA500", "#FF0000"]
#
# bar_width = (2 * np.pi) / (len(categories) * len(data_rpi1807) + len(data_rpi1807))
#
# for idx, (feature_data, color) in enumerate(zip(data_rpi1807, colors)):
#     data = feature_data + feature_data[:1]  # Complete the loop
#     adjusted_angles = [
#         angle - (bar_width * (len(data_rpi1807) - 1)) / 2 + idx * bar_width
#         for angle in angles[:-1]
#     ] + [
#         angles[0] - (bar_width * (len(data_rpi1807) - 1)) / 2 + idx * bar_width
#     ]
#     ax.bar(
#         adjusted_angles,
#         data,
#         color=color,
#         width=bar_width,
#         label=feature_names[idx],
#         alpha=0.85,
#         zorder=1,
#     )
#     for angle, value in zip(adjusted_angles, feature_data):
#         angle_deg = (np.degrees(np.pi / 2 - angle)) % 360
#         if 90 < angle_deg < 270:
#             rotation = angle_deg + 180
#             r = value - 11
#             alignment = 'right'
#         else:
#             rotation = angle_deg
#             r = value - 11
#             alignment = 'left'
#         ax.text(
#             angle,
#             r,
#             f"{value:.1f}",
#             color="black",
#             size=14,
#             ha=alignment,
#             va="center",
#             rotation=rotation,
#             rotation_mode='anchor',
#             zorder=4,
#         )
#
# ax.xaxis.grid(True, color="gray", linestyle="-", linewidth=1.0, zorder=1)
# ax.yaxis.grid(True, color="gray", linestyle="--", linewidth=0.5, zorder=1)
# ax.yaxis.set_tick_params(labelsize=14)
#
# ax.set_theta_offset(np.pi / 2)
# ax.set_theta_direction(-1)
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(categories, fontsize=16)
# ax.set_title("RPI1807", fontsize=20)
# ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1), fontsize=16, frameon=False)
# plt.tight_layout()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
#
# # Data for RPI369 and RPI488
# data_rpi369 = [
#     [50.2, 23.7, 77.1, 51.2, 0.9, 46.8],
#     [71.3, 71.6, 70.2, 72.4, 42.6, 71.3],
#     [70.0, 78.4, 56.0, 84.0, 42.8, 70.0],
#     [72.8, 79.7, 65.9, 70.1, 46.1, 82.1],
#     [80.1, 82.1, 78.0, 79.2, 60.4, 86.2],
# ]
# data_rpi488 = [
#     [85.6, 77.0, 94.7, 94.0, 72.5, 92.9],
#     [88.3, 92.8, 83.1, 93.5, 77.1, 88.3],
#     [89.3, 94.6, 83.5, 95.1, 79.3, 89.3],
#     [89.3, 83.9, 94.7, 94.3, 79.3, 91.1],
#     [89.3, 83.5, 95.1, 94.6, 79.3, 91.6],
# ]
#
# categories = ['ACC', 'SEN', 'SPE', 'PRE', 'MCC', 'AUC']
# feature_names = [
#     "lncPro",
#     "RPISeq-RF",
#     "IPMiner",
#     "RPITER",
#     "Transfer-RPI",
# ]
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={"projection": "polar"})
#
# # Calculate angles
# angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
# angles += angles[:1]
#
# colors = ["#6699FF", "#FF6666", "#ADFF2F", "#00BFFF", "#FF0000"]
#
# bar_width = (2 * np.pi) / (len(categories) * len(data_rpi369) + len(data_rpi369))
#
# for idx, (feature_data, color) in enumerate(zip(data_rpi369, colors)):
#     data = feature_data + feature_data[:1]  # Complete the loop
#     adjusted_angles = [
#         angle - (bar_width * (len(data_rpi369) - 1)) / 2 + idx * bar_width
#         for angle in angles[:-1]
#     ] + [
#         angles[0] - (bar_width * (len(data_rpi369) - 1)) / 2 + idx * bar_width
#     ]
#     ax1.bar(
#         adjusted_angles,
#         data,
#         color=color,
#         width=bar_width,
#         label=feature_names[idx],
#         alpha=0.85,
#         zorder=1,
#     )
#     for angle, value in zip(adjusted_angles, feature_data):
#         angle_deg = (np.degrees(np.pi / 2 - angle)) % 360
#         if 90 < angle_deg < 270:
#             rotation = angle_deg + 180
#             r = value - 12
#             alignment = 'right'
#         else:
#             rotation = angle_deg
#             r = value - 12
#             alignment = 'left'
#         ax1.text(
#             angle,
#             r,
#             f"{value:.1f}",
#             color="black",
#             size=14,
#             ha=alignment,
#             va="center",
#             rotation=rotation,
#             rotation_mode='anchor',
#             zorder=4,
#         )
#
# ax1.xaxis.grid(True, color="gray", linestyle="-", linewidth=1.0, zorder=1)  # Solid line
# ax1.yaxis.grid(True, color="gray", linestyle="--", linewidth=0.5, zorder=1)  # Dashed line
# ax1.yaxis.set_tick_params(labelsize=14)
# ax1.set_theta_offset(np.pi / 2)
# ax1.set_theta_direction(-1)
# ax1.set_xticks(angles[:-1])
# ax1.set_xticklabels(categories, fontsize=16)
# ax1.set_title("RPI369", fontsize=20)
#
# for idx, (feature_data, color) in enumerate(zip(data_rpi488, colors)):
#     data = feature_data + feature_data[:1]  # Complete the loop
#     adjusted_angles = [
#         angle - (bar_width * (len(data_rpi488) - 1)) / 2 + idx * bar_width
#         for angle in angles[:-1]
#     ] + [
#         angles[0] - (bar_width * (len(data_rpi488) - 1)) / 2 + idx * bar_width
#     ]
#     ax2.bar(
#         adjusted_angles,
#         data,
#         color=color,
#         width=bar_width,
#         label=feature_names[idx],
#         alpha=0.85,
#         zorder=1,
#     )
#     for angle, value in zip(adjusted_angles, feature_data):
#         angle_deg = (np.degrees(np.pi / 2 - angle)) % 360
#         if 90 < angle_deg < 270:
#             rotation = angle_deg + 180
#             r = value - 12
#             alignment = 'right'
#         else:
#             rotation = angle_deg
#             r = value - 12
#             alignment = 'left'
#         ax2.text(
#             angle,
#             r,
#             f"{value:.1f}",
#             color="black",
#             size=14,
#             ha=alignment,
#             va="center",
#             rotation=rotation,
#             rotation_mode='anchor',
#             zorder=4,
#         )
#
# ax2.xaxis.grid(True, color="gray", linestyle="-", linewidth=1.0, zorder=1)
# ax2.yaxis.grid(True, color="gray", linestyle="--", linewidth=0.5, zorder=1)
# ax2.yaxis.set_tick_params(labelsize=14)
# ax2.set_theta_offset(np.pi / 2)
# ax2.set_theta_direction(-1)
# ax2.set_xticks(angles[:-1])
# ax2.set_xticklabels(categories, fontsize=16)
# ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1), fontsize=16, frameon=False)
# ax2.set_title("RPI488", fontsize=20)
#
# plt.tight_layout()
# plt.show()

