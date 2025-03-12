import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch, Rectangle

# 加载 RPI2241 数据集
labels1 = np.load('../data/ROC_label/RPI2241/labels.npy')
RiNALMo_PAAC_preds1 = np.load('../data/ROC_label/RPI2241/RiNALMo_PAAC.npy')
RiNALMo_ESM_preds1 = np.load('../data/ROC_label/RPI2241/RiNALMo_ESM.npy')
kmer_PAAC_preds1 = np.load('../data/ROC_label/RPI2241/k-mer_PAAC.npy')
kmer_ESM_preds1 = np.load('../data/ROC_label/RPI2241/k-mer_ESM.npy')

# 加载 NPInter 数据集
labels2 = np.load('../data/ROC_label/NPInter/labels.npy')
RiNALMo_PAAC_preds2 = np.load('../data/ROC_label/NPInter/RiNALMo_PAAC.npy')
RiNALMo_ESM_preds2 = np.load('../data/ROC_label/NPInter/RiNALMo_ESM.npy')
kmer_PAAC_preds2 = np.load('../data/ROC_label/NPInter/k-mer_PAAC.npy')
kmer_ESM_preds2 = np.load('../data/ROC_label/NPInter/k-mer_ESM.npy')


# 计算 RPI2241 的 ROC 曲线和 AUC
fpr1_1, tpr1_1, _ = roc_curve(labels1, RiNALMo_PAAC_preds1)
roc_auc1_1 = auc(fpr1_1, tpr1_1)
fpr1_2, tpr1_2, _ = roc_curve(labels1, RiNALMo_ESM_preds1)
roc_auc1_2 = auc(fpr1_2, tpr1_2)
fpr1_3, tpr1_3, _ = roc_curve(labels1, kmer_PAAC_preds1)
roc_auc1_3 = auc(fpr1_3, tpr1_3)
fpr1_4, tpr1_4, _ = roc_curve(labels1, kmer_ESM_preds1)
roc_auc1_4 = auc(fpr1_4, tpr1_4)

# 计算 NPInter 的 ROC 曲线和 AUC
fpr2_1, tpr2_1, _ = roc_curve(labels2, RiNALMo_PAAC_preds2)
roc_auc2_1 = auc(fpr2_1, tpr2_1)
fpr2_2, tpr2_2, _ = roc_curve(labels2, RiNALMo_ESM_preds2)
roc_auc2_2 = auc(fpr2_2, tpr2_2)
fpr2_3, tpr2_3, _ = roc_curve(labels2, kmer_PAAC_preds2)
roc_auc2_3 = auc(fpr2_3, tpr2_3)
fpr2_4, tpr2_4, _ = roc_curve(labels2, kmer_ESM_preds2)
roc_auc2_4 = auc(fpr2_4, tpr2_4)

# 创建两个并排的子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# 绘制 RPI2241 数据集的 ROC 曲线
ax1.plot(fpr1_2, tpr1_2, color='red', lw=2, label='RiNALMo+ESM (AUC = %0.3f)' % roc_auc1_2)
ax1.plot(fpr1_1, tpr1_1, color='orange', lw=2, label='RiNALMo+PAAC (AUC = %0.3f)' % roc_auc1_1)
ax1.plot(fpr1_4, tpr1_4, color='green', lw=2, label='k-mer+ESM (AUC = %0.3f)' % roc_auc1_4)
ax1.plot(fpr1_3, tpr1_3, color='blue', lw=2, label='k-mer+PAAC (AUC = %0.3f)' % roc_auc1_3)
ax1.plot([0, 1], [0, 1], color='cyan', lw=2, linestyle='--')

# 添加 RPI2241 数据集的标签和标题
ax1.set_xlim([-0.05, 1.05])
ax1.set_ylim([-0.05, 1.05])
ax1.set_xlabel('False Positive Rate', fontsize=16)
ax1.set_ylabel('True Positive Rate', fontsize=16)
ax1.set_title('RPI369 ROC and AUC', fontsize=20)
ax1.legend(loc="lower right")

# 添加嵌套子图到 RPI2241 的图中
axins1 = inset_axes(ax1, width="100%", height="100%", loc='center',
                   bbox_to_anchor=(0.4, 0.4, 0.56, 0.35), bbox_transform=ax1.transAxes)
axins1.plot(fpr1_2, tpr1_2, color='red', lw=2)
axins1.plot(fpr1_1, tpr1_1, color='orange', lw=2)
axins1.plot(fpr1_4, tpr1_4, color='green', lw=2)
axins1.plot(fpr1_3, tpr1_3, color='blue', lw=2)
axins1.set_xlim(0.0, 0.3)
axins1.set_ylim(0.8, 1.0)
for spine in axins1.spines.values():
    spine.set_edgecolor('purple')
    spine.set_linewidth(2)

axins1.set_xticks([])  # 隐藏x轴刻度
axins1.set_yticks([])  # 隐藏y轴刻度
ax1.legend(fontsize=16)
ax1.tick_params(axis='both', labelsize=14)
# 添加放大区域的框
rect1 = Rectangle((0.0, 0.8), 0.3, 0.2, fill=False, edgecolor='purple', linestyle='dashed', linewidth=2)
ax1.add_patch(rect1)
con1 = ConnectionPatch(xyA=(0.3, 0.8), coordsA=ax1.transData,
                      xyB=(0.0, 1.0), coordsB=axins1.transData,
                      arrowstyle="->", color="purple", lw=2)
ax1.add_artist(con1)

# 绘制 NPInter 数据集的 ROC 曲线
ax2.plot(fpr2_2, tpr2_2, color='red', lw=2, label='RiNALMo+ESM (AUC = %0.3f)' % roc_auc2_2)
ax2.plot(fpr2_1, tpr2_1, color='orange', lw=2, label='RiNALMo+PAAC (AUC = %0.3f)' % roc_auc2_1)
ax2.plot(fpr2_4, tpr2_4, color='green', lw=2, label='k-mer+ESM (AUC = %0.3f)' % roc_auc2_4)
ax2.plot(fpr2_3, tpr2_3, color='blue', lw=2, label='k-mer+PAAC (AUC = %0.3f)' % roc_auc2_3)
ax2.plot([0, 1], [0, 1], color='cyan', lw=2, linestyle='--')

# 添加 NPInter 数据集的标签和标题
ax2.set_xlim([-0.05, 1.05])
ax2.set_ylim([-0.05, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=16)
ax2.set_ylabel('True Positive Rate', fontsize=16)
ax2.set_title('RPI488 ROC and AUC', fontsize=20)
ax2.legend(loc="lower right")
ax2.tick_params(axis='both', labelsize=14)
# 添加嵌套子图到 NPInter 的图中
axins2 = inset_axes(ax2, width="100%", height="100%", loc='center',
                   bbox_to_anchor=(0.4, 0.4, 0.56, 0.35), bbox_transform=ax2.transAxes)
axins2.plot(fpr2_2, tpr2_2, color='red', lw=2)
axins2.plot(fpr2_1, tpr2_1, color='orange', lw=2)
axins2.plot(fpr2_4, tpr2_4, color='green', lw=2)
axins2.plot(fpr2_3, tpr2_3, color='blue', lw=2)
axins2.set_xlim(0.0, 0.3)
axins2.set_ylim(0.8, 1.0)
for spine in axins2.spines.values():
    spine.set_edgecolor('purple')
    spine.set_linewidth(2)

axins2.set_xticks([])  # 隐藏x轴刻度
axins2.set_yticks([])  # 隐藏y轴刻度
ax2.legend(fontsize=16)

# 添加放大区域的框
rect2 = Rectangle((0.0, 0.8), 0.3, 0.2, fill=False, edgecolor='purple', linestyle='dashed', linewidth=2)
ax2.add_patch(rect2)
con2 = ConnectionPatch(xyA=(0.3, 0.8), coordsA=ax2.transData,
                      xyB=(0.0, 1.0), coordsB=axins2.transData,
                      arrowstyle="->", color="purple", lw=2)
ax2.add_artist(con2)

plt.tight_layout()
plt.show()




# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# import numpy as np
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib.patches import ConnectionPatch, Rectangle
# # 加载数据
#
# labels = np.load('../data/ROC_label/RPI1807/labels.npy')
# RiNALMo_PAAC_preds = np.load('../data/ROC_label/RPI1807/RiNALMo_PAAC.npy')
# RiNALMo_ESM_preds = np.load('../data/ROC_label/RPI1807/RiNALMo_ESM.npy')
# kmer_PAAC_preds = np.load('../data/ROC_label/RPI1807/k-mer_PAAC.npy')
# kmer_ESM_preds = np.load('../data/ROC_label/RPI1807/k-mer_ESM.npy')
#
# # 计算每个模型的ROC曲线和AUC
# fpr1, tpr1, _ = roc_curve(labels, RiNALMo_PAAC_preds)
# roc_auc1 = auc(fpr1, tpr1)
# fpr2, tpr2, _ = roc_curve(labels, RiNALMo_ESM_preds)
# roc_auc2 = auc(fpr2, tpr2)
# fpr3, tpr3, _ = roc_curve(labels, kmer_PAAC_preds)
# roc_auc3 = auc(fpr3, tpr3)
# fpr4, tpr4, _ = roc_curve(labels, kmer_ESM_preds)
# roc_auc4 = auc(fpr4, tpr4)
#
# # 创建基本图像
# fig, ax = plt.subplots()
# ax.plot(fpr2, tpr2, color='red', lw=2, label='RiNALMo+ESM (AUC = %0.3f)' % roc_auc2)
# ax.plot(fpr1, tpr1, color='orange', lw=2, label='RiNALMo+PAAC (AUC = %0.3f)' % roc_auc1)
# ax.plot(fpr4, tpr4, color='green', lw=2, label='k-mer+ESM (AUC = %0.3f)' % roc_auc4)
# ax.plot(fpr3, tpr3, color='blue', lw=2, label='k-mer+PAAC (AUC = %0.3f)' % roc_auc3)
# ax.plot([0, 1], [0, 1], color='cyan', lw=2, linestyle='--')
#
# # 添加标签和标题
# ax.set_xlim([-0.05, 1.05])
# ax.set_ylim([-0.05, 1.05])
# ax.set_xlabel('False Positive Rate')
# ax.set_ylabel('True Positive Rate')
# ax.set_title('RPI1807 ROC and AUC')
# ax.legend(loc="lower right")
#
# axins = inset_axes(ax, width="100%", height="100%", loc='center',
#                    bbox_to_anchor=(0.4, 0.4, 0.56, 0.35), bbox_transform=ax.transAxes)
# axins.plot(fpr2, tpr2, color='red', lw=2)
# axins.plot(fpr1, tpr1, color='orange', lw=2)
# axins.plot(fpr4, tpr4, color='green', lw=2)
# axins.plot(fpr3, tpr3, color='blue', lw=2)
#
# # # 设置放大区域的范围
# axins.set_xlim(0.0, 0.3)
# axins.set_ylim(0.8, 1.0)
# for spine in axins.spines.values():
#     spine.set_edgecolor('purple')
#     spine.set_linewidth(2)
#
# axins.set_xticks([])  # 隐藏x轴刻度
# axins.set_yticks([])  # 隐藏y轴刻度
#
# # 添加主图中被放大区域的框
# rect = Rectangle((0.0, 0.8), 0.3, 0.2, fill=False, edgecolor='purple', linestyle='dashed', linewidth=2)
# ax.add_patch(rect)
#
# con = ConnectionPatch(xyA=(0.3, 0.8), coordsA=ax.transData,
#                       xyB=(0.0, 1.0), coordsB=axins.transData,
#                       arrowstyle="->", color="purple", lw=2)
# ax.add_artist(con)
# plt.show()







# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# import numpy as np
#
#
# labels1 = np.load('../data/ROC_label/RPI369/labels.npy')
# RiNALMo_PAAC_preds1 = np.load('../data/ROC_label/RPI369/RiNALMo_PAAC.npy')
# RiNALMo_ESM_preds1 = np.load('../data/ROC_label/RPI369/RiNALMo_ESM.npy')
# kmer_PAAC_preds1 = np.load('../data/ROC_label/RPI369/k-mer_PAAC.npy')
# kmer_ESM_preds1 = np.load('../data/ROC_label/RPI369/k-mer_ESM.npy')
#
# # 加载 NPInter 数据集
# labels2 = np.load('../data/ROC_label/RPI488/labels.npy')
# RiNALMo_PAAC_preds2 = np.load('../data/ROC_label/RPI488/RiNALMo_PAAC.npy')
# RiNALMo_ESM_preds2 = np.load('../data/ROC_label/RPI488/RiNALMo_ESM.npy')
# kmer_PAAC_preds2 = np.load('../data/ROC_label/RPI488/k-mer_PAAC.npy')
# kmer_ESM_preds2 = np.load('../data/ROC_label/RPI488/k-mer_ESM.npy')
#
# fpr1_1, tpr1_1, _ = roc_curve(labels1, RiNALMo_PAAC_preds1)
# roc_auc1_1 = auc(fpr1_1, tpr1_1)
# fpr1_2, tpr1_2, _ = roc_curve(labels1, RiNALMo_ESM_preds1)
# roc_auc1_2 = auc(fpr1_2, tpr1_2)
# fpr1_3, tpr1_3, _ = roc_curve(labels1, kmer_PAAC_preds1)
# roc_auc1_3 = auc(fpr1_3, tpr1_3)
# fpr1_4, tpr1_4, _ = roc_curve(labels1, kmer_ESM_preds1)
# roc_auc1_4 = auc(fpr1_4, tpr1_4)
#
# fpr2_1, tpr2_1, _ = roc_curve(labels2, RiNALMo_PAAC_preds2)
# roc_auc2_1 = auc(fpr2_1, tpr2_1)
# fpr2_2, tpr2_2, _ = roc_curve(labels2, RiNALMo_ESM_preds2)
# roc_auc2_2 = auc(fpr2_2, tpr2_2)
# fpr2_3, tpr2_3, _ = roc_curve(labels2, kmer_PAAC_preds2)
# roc_auc2_3 = auc(fpr2_3, tpr2_3)
# fpr2_4, tpr2_4, _ = roc_curve(labels2, kmer_ESM_preds2)
# roc_auc2_4 = auc(fpr2_4, tpr2_4)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
#
# ax1.plot(fpr1_2, tpr1_2, color='red', lw=2, label='RiNALMo+ESM (AUC = %0.3f)' % roc_auc1_2)
# ax1.plot(fpr1_1, tpr1_1, color='orange', lw=2, label='RiNALMo+PAAC (AUC = %0.3f)' % roc_auc1_1)
# ax1.plot(fpr1_4, tpr1_4, color='green', lw=2, label='k-mer+ESM (AUC = %0.3f)' % roc_auc1_4)
# ax1.plot(fpr1_3, tpr1_3, color='blue', lw=2, label='k-mer+PAAC (AUC = %0.3f)' % roc_auc1_3)
# ax1.plot([0, 1], [0, 1], color='cyan', lw=2, linestyle='--')
#
# ax1.set_xlim([-0.05, 1.05])
# ax1.set_ylim([-0.05, 1.05])
# ax1.set_xlabel('False Positive Rate', fontsize=16)
# ax1.set_ylabel('True Positive Rate', fontsize=16)
# ax1.set_title('RPI369 ROC and AUC', fontsize=20)
# ax1.legend(loc="lower right")
#
# ax1.legend(fontsize=16)
# ax1.tick_params(axis='both', labelsize=14)  # 刻度字体
#
#
#
# ax2.plot(fpr2_2, tpr2_2, color='red', lw=2, label='RiNALMo+ESM (AUC = %0.3f)' % roc_auc2_2)
# ax2.plot(fpr2_1, tpr2_1, color='orange', lw=2, label='RiNALMo+PAAC (AUC = %0.3f)' % roc_auc2_1)
# ax2.plot(fpr2_4, tpr2_4, color='green', lw=2, label='k-mer+ESM (AUC = %0.3f)' % roc_auc2_4)
# ax2.plot(fpr2_3, tpr2_3, color='blue', lw=2, label='k-mer+PAAC (AUC = %0.3f)' % roc_auc2_3)
# ax2.plot([0, 1], [0, 1], color='cyan', lw=2, linestyle='--')
#
#
# ax2.set_xlim([-0.05, 1.05])
# ax2.set_ylim([-0.05, 1.05])
# ax2.set_xlabel('False Positive Rate', fontsize=16)
# ax2.set_ylabel('True Positive Rate', fontsize=16)
# ax2.set_title('RPI488 ROC and AUC', fontsize=20)
# ax2.legend(loc="lower right")
# ax2.tick_params(axis='both', labelsize=14)
# ax2.legend(fontsize=16)
# plt.tight_layout()
# plt.show()