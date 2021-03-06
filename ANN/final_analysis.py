import os
import argparse

from pathlib import Path

import matplotlib.pyplot as plt


def main():
    # Create folder for report
    Path("../Analysis/Final/Best").mkdir(exist_ok=True, parents=True)
    Path("../Analysis/Final/Averages").mkdir(exist_ok=True)
    Path("../Analysis/Final/Averages_comp").mkdir(exist_ok=True)
    Path("../Analysis/Final/PR_curves").mkdir(exist_ok=True)
    Path("../Analysis/Final/ROC_curves").mkdir(exist_ok=True)
    Path("../Analysis/Final/Fscore_curves").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--step4_folder', default='default', help='Best step4 models log folder')
    parser.add_argument('--step5_folder', default='default', help='Step5 log folder')
    parser.add_argument('--avg_folder', default='default', help='Average models log folder')
    parser.add_argument('--avg_das_folder', default='default', help='Average models das log folder')
    parser.add_argument('--best_models', default='', help='Best model names space separated')
    parser.add_argument('--avg_models', default='', help='Average model names space separated')
    parser.add_argument('--n_thresh_step4', type=int, default=19, help='Number of thresholds evaluated')
    parser.add_argument('--n_thresh_step5', type=int, default=29, help='Number of thresholds evaluated')
    args = parser.parse_args()

    # Comparar mejores modelos con modelos promedio
    # working directory
    step4_eval_wkdir = '../Analysis/logs/eval/' + args.step4_folder
    step5_eval_wkdir = '../Analysis/logs/eval/' + args.step5_folder
    avg_eval_wkdir = '../Analysis/logs/eval/' + args.avg_folder
    avg_eval_das_wkdir = '../Analysis/logs/eval/' + args.avg_das_folder

    best_models = args.best_models.strip().split(' ')
    avg_models = args.avg_models.strip().split(' ')

    # Metrics
    thresholds = []

    step4_pre = []
    step4_rec = []
    step4_fpr = []
    step4_fsc = []

    step5_pre = []
    step5_rec = []
    step5_fpr = []
    step5_fsc = []

    avg_pre = []
    avg_rec = []
    avg_fpr = []
    avg_fsc = []

    avg_das_pre = []
    avg_das_rec = []
    avg_das_fpr = []
    avg_das_fsc = []

    # Lists of PR, ROC and Fscore curves
    step4_pr_curves = []
    step4_roc_curves = []
    step4_fscore_curves = []

    step5_pr_curves = []
    step5_roc_curves = []
    step5_fscore_curves = []

    avg_pr_curves = []
    avg_roc_curves = []
    avg_fscore_curves = []

    avg_das_pr_curves = []
    avg_das_roc_curves = []
    avg_das_fscore_curves = []

    # PR and ROC AUCs
    step4_pr_aucs = []
    step4_roc_aucs = []

    step5_pr_aucs = []
    step5_roc_aucs = []

    avg_pr_aucs = []
    avg_roc_aucs = []

    avg_das_pr_aucs = []
    avg_das_roc_aucs = []

    # Best Fscores
    avg_best_fscores = []
    avg_das_best_fscores = []
    step4_best_fscores = []
    step5_best_fscores = []

    # Read average model variables
    for f_name in avg_models:
        with open(os.path.join(avg_eval_wkdir, f_name), 'r') as f:

            f.readline()
            f.readline()

            # for _ in range(args.n_thresh_step4):
            for _ in range(39):

                thresh = f.readline().split(':')[-1].strip()
                thresholds.append(thresh)

                # Skip non-useful lines
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                # acc
                f.readline()

                # Read metrics
                avg_pre.append(f.readline().split(":")[-1].strip())
                avg_rec.append(f.readline().split(":")[-1].strip())
                avg_fpr.append(f.readline().split(":")[-1].strip())
                avg_fsc.append(f.readline().split(":")[-1].strip())

                f.readline()
                f.readline()
                f.readline()

            # Terminar de leer el archivo
            avg_best_fscores.append(f.readline().split(",")[-1].strip().split(":")[-1].strip())

            f.readline()

            avg_pr_aucs.append(f.readline().split(":")[-1].strip())
            avg_roc_aucs.append(f.readline().split(":")[-1].strip())

            avg_pre = list(map(float, avg_pre))
            avg_rec = list(map(float, avg_rec))
            avg_fpr = list(map(float, avg_fpr))
            avg_fsc = list(map(float, avg_fsc))
            thresholds = list(map(float, thresholds))

            # Aqui armar la curva y agregarlas a la lista mayor
            avg_pr_curves.append([avg_rec, avg_pre])
            avg_roc_curves.append([avg_fpr, avg_rec])
            avg_fscore_curves.append([thresholds, avg_fsc])

            avg_pre = []
            avg_rec = []
            avg_fpr = []
            avg_fsc = []
            thresholds = []

    avg_best_fscores = list(map(float, avg_best_fscores))
    avg_pr_aucs = list(map(float, avg_pr_aucs))
    avg_roc_aucs = list(map(float, avg_roc_aucs))

    # Read average model variables
    for f_name in avg_models:
        with open(os.path.join(avg_eval_das_wkdir, f_name), 'r') as f:

            for _ in range(args.n_thresh_step4):
                thresh = f.readline().split(':')[-1].strip()
                thresholds.append(thresh)

                # Space after threshold
                f.readline()

                # True positives
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                # Space
                f.readline()

                # True negatives
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                # Space
                f.readline()

                # Total seismic, non-seismic
                f.readline()
                f.readline()

                # Space
                f.readline()

                # Total TP, TN, FP, FN
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                # Space and Accuracy
                f.readline()
                f.readline()

                # Read metrics
                avg_das_pre.append(f.readline().split(":")[-1].strip())
                avg_das_rec.append(f.readline().split(":")[-1].strip())
                avg_das_fpr.append(f.readline().split(":")[-1].strip())
                avg_das_fsc.append(f.readline().split(":")[-1].strip())

                f.readline()

            # Terminar de leer el archivo
            avg_das_best_fscores.append(f.readline().split(",")[-1].strip().split(":")[-1].strip())
            avg_das_pr_aucs.append(f.readline().split(":")[-1].strip())
            avg_das_roc_aucs.append(f.readline().split(":")[-1].strip())

            avg_das_pre = list(map(float, avg_das_pre))
            avg_das_rec = list(map(float, avg_das_rec))
            avg_das_fpr = list(map(float, avg_das_fpr))
            avg_das_fsc = list(map(float, avg_das_fsc))
            thresholds = list(map(float, thresholds))

            # Aqui armar la curva y agregarlas a la lista mayor
            avg_das_pr_curves.append([avg_das_rec, avg_das_pre])
            avg_das_roc_curves.append([avg_das_fpr, avg_das_rec])
            avg_das_fscore_curves.append([thresholds, avg_das_fsc])

            avg_das_pre = []
            avg_das_rec = []
            avg_das_fpr = []
            avg_das_fsc = []
            thresholds = []

    avg_das_best_fscores = list(map(float, avg_das_best_fscores))
    avg_das_pr_aucs = list(map(float, avg_das_pr_aucs))
    avg_das_roc_aucs = list(map(float, avg_das_roc_aucs))

    for f_name in best_models:
        with open(os.path.join(step4_eval_wkdir, f_name), 'r') as f:

            f.readline()
            f.readline()

            # for _ in range(args.n_thresh_step4):
            for _ in range(39):

                thresh = f.readline().split(':')[-1].strip()
                thresholds.append(thresh)

                # Skip non-useful lines
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                # acc
                f.readline()

                # Read metrics
                step4_pre.append(f.readline().split(":")[-1].strip())
                step4_rec.append(f.readline().split(":")[-1].strip())
                step4_fpr.append(f.readline().split(":")[-1].strip())
                step4_fsc.append(f.readline().split(":")[-1].strip())

                f.readline()
                f.readline()
                f.readline()

            # Terminar de leer el archivo
            step4_best_fscores.append(f.readline().split(",")[-1].strip().split(":")[-1].strip())

            f.readline()

            step4_pr_aucs.append(f.readline().split(":")[-1].strip())
            step4_roc_aucs.append(f.readline().split(":")[-1].strip())

            # print(f'rec: {rec}\npre: {pre}\nfpr: {fpr}\nfsc: {fsc}')

            step4_pre = list(map(float, step4_pre))
            step4_rec = list(map(float, step4_rec))
            step4_fpr = list(map(float, step4_fpr))
            step4_fsc = list(map(float, step4_fsc))
            thresholds = list(map(float, thresholds))

            # Aqui armar la curva y agregarlas a la lista mayor
            step4_pr_curves.append([step4_rec, step4_pre])
            step4_roc_curves.append([step4_fpr, step4_rec])
            step4_fscore_curves.append([thresholds, step4_fsc])

            step4_pre = []
            step4_rec = []
            step4_fpr = []
            step4_fsc = []
            thresholds = []

    step4_best_fscores = list(map(float, step4_best_fscores))
    step4_pr_aucs = list(map(float, step4_pr_aucs))
    step4_roc_aucs = list(map(float, step4_roc_aucs))

    for f_name in best_models:
        with open(os.path.join(step5_eval_wkdir, f_name), 'r') as f:

            for _ in range(args.n_thresh_step5):

                thresh = f.readline().split(':')[-1].strip()
                thresholds.append(thresh)

                # Space after threshold
                f.readline()

                # True positives
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                # Space
                f.readline()

                # True negatives
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                # Space
                f.readline()

                # Total seismic, non-seismic
                f.readline()
                f.readline()

                # Space
                f.readline()

                # Total TP, TN, FP, FN
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                # Space and Accuracy
                f.readline()
                f.readline()

                # Read metrics
                step5_pre.append(f.readline().split(":")[-1].strip())
                step5_rec.append(f.readline().split(":")[-1].strip())
                step5_fpr.append(f.readline().split(":")[-1].strip())
                step5_fsc.append(f.readline().split(":")[-1].strip())

                f.readline()

            # Terminar de leer el archivo
            step5_best_fscores.append(f.readline().split(",")[-1].strip().split(":")[-1].strip())
            step5_pr_aucs.append(f.readline().split(":")[-1].strip())
            step5_roc_aucs.append(f.readline().split(":")[-1].strip())

            step5_pre = list(map(float, step5_pre))
            step5_rec = list(map(float, step5_rec))
            step5_fpr = list(map(float, step5_fpr))
            step5_fsc = list(map(float, step5_fsc))
            thresholds = list(map(float, thresholds))

            # Aqui armar la curva y agregarlas a la lista mayor
            step5_pr_curves.append([step5_rec, step5_pre])
            step5_roc_curves.append([step5_fpr, step5_rec])
            step5_fscore_curves.append([thresholds, step5_fsc])

            step5_pre = []
            step5_rec = []
            step5_fpr = []
            step5_fsc = []
            thresholds = []

    step5_best_fscores = list(map(float, step5_best_fscores))
    step5_pr_aucs = list(map(float, step5_pr_aucs))
    step5_roc_aucs = list(map(float, step5_roc_aucs))

    # Mejores curvas step4

    # Curvas PR
    plt.clf()

    for crv, mdl in zip(step4_pr_curves, best_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    # Dumb model line
    plt.hlines(0.5, 0, 1, 'b', '--')
    plt.title(f'PR curves best models ANN STEAD')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(-0.02, 1.02)
    plt.ylim(0.48, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Best/best_PR_ann.png')
    plt.xlim(0.92, 1.01)
    plt.ylim(0.88, 1.01)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Best/best_PR_ann_zoom.png')

    # Curva ROC
    plt.clf()

    for crv, mdl in zip(step4_roc_curves, best_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    # Dumb model line
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC curves best models ANN STEAD')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Best/best_ROC_ann.png')
    plt.xlim(-0.01, 0.1)
    plt.ylim(0.88, 1)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Best/best_ROC_ann_zoom.png')

    # Curva Fscore
    plt.clf()

    for crv, mdl in zip(step4_fscore_curves, best_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    plt.title(f'Fscore vs thresholds curves best models ANN STEAD')
    plt.xlabel('Threshold')
    plt.ylabel('F-score')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Best/best_Fscore_ann.png')
    plt.xlim(0, 0.5)
    plt.ylim(-0.02, 1)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Best/best_Fscore_ann_zoom.png')

    # Curvas STEAD Y DAS step4 vs step5 mejores modelos

    # Curvas PR
    plt.figure()

    ax = plt.gca()

    for crv4, crv5, mdl in zip(step4_pr_curves, step5_pr_curves, best_models):
        axcolor = next(ax._get_lines.prop_cycler)['color']
        plt.plot(crv4[0], crv4[1], marker='.', markersize=6, label=mdl.strip().split('.')[0], color=axcolor)
        plt.plot(crv5[0], crv5[1], marker='.', markersize=6, color=axcolor)

    # Dumb model line
    plt.hlines(0.5, 0, 1, 'b', '--')
    plt.title(f'PR curves best models ANN STEAD y DAS')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(-0.02, 1.02)
    plt.ylim(0.48, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Best/Comp_PR_ANN.png')
    plt.xlim(-0.02, 1.02)
    plt.ylim(0, 1.02)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Best/Comp_PR_ANN_full.png')

    # Curva ROC
    plt.clf()

    for crv4, crv5, mdl in zip(step4_roc_curves, step5_roc_curves, best_models):
        axcolor = next(ax._get_lines.prop_cycler)['color']
        plt.plot(crv4[0], crv4[1], marker='.', markersize=6, label=mdl.strip().split('.')[0], color=axcolor)
        plt.plot(crv5[0], crv5[1], marker='.', markersize=6, color=axcolor)

    # Dumb model line
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC curves best models ANN STEAD y DAS')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Best/Comp_ROC_ANN.png')

    # Curva Fscore
    plt.clf()

    for crv4, crv5, mdl in zip(step4_fscore_curves, step5_fscore_curves, best_models):
        axcolor = next(ax._get_lines.prop_cycler)['color']
        plt.plot(crv4[0], crv4[1], marker='.', markersize=6, label=mdl.strip().split('.')[0], color=axcolor)
        plt.plot(crv5[0], crv5[1], marker='.', markersize=6, color=axcolor)

    plt.title(f'Fscore vs thresholds curves best models ANN STEAD y DAS')
    plt.xlabel('Threshold')
    plt.ylabel('F-score')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Best/Comp_Fscore_ANN.png')

    plt.xlim(0, 0.5)
    plt.ylim(-0.02, 1.02)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Best/Comp_Fscore_ANN_zoom.png')

    # Comparacion mejores curvas step4 y modelos promedio

    # Curvas PR
    plt.clf()

    for crv, mdl in zip(avg_pr_curves, avg_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    for crv, mdl in zip(step4_pr_curves, best_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    # Dumb model line
    plt.hlines(0.5, 0, 1, 'b', '--')
    plt.title(f'ANN best and average models PR curve comparison')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(-0.02, 1.02)
    plt.ylim(0.48, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False, ncol=2)
    plt.savefig(f'../Analysis/Final/Averages/comp_best_avg_PR_ann.png')

    # Curva ROC
    plt.clf()

    for crv, mdl in zip(avg_roc_curves, avg_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    for crv, mdl in zip(step4_roc_curves, best_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    # Dumb model line
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ANN best and average models ROC curve comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Averages/comp_best_avg_ROC_ann.png')

    # Curva Fscore
    plt.clf()

    for crv, mdl in zip(avg_fscore_curves, avg_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    for crv, mdl in zip(step4_fscore_curves, best_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    plt.title(f'ANN best and average models Fscore curve comparison')
    plt.xlabel('Threshold')
    plt.ylabel('F-score')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Averages/comp_best_avg_Fscore_ann.png')

    # Comparación curvas step5 mejores modelos y promedio

    # Curva PR
    plt.clf()

    for crv, mdl in zip(step5_pr_curves, best_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    for crv, mdl in zip(avg_das_pr_curves, avg_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    # Dumb model line
    plt.hlines(0.5, 0, 1, 'b', '--')
    plt.title(f'PR curves best models and average models ANN')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(-0.02, 1.02)
    plt.ylim(0.48, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False, ncol=2)
    plt.savefig(f'../Analysis/Final/Averages_comp/avgcomp_PR_ann.png')

    # Curva ROC
    plt.clf()

    for crv, mdl in zip(step5_roc_curves, best_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    for crv, mdl in zip(avg_das_roc_curves, avg_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    # Dumb model line
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC curves best models and average models ANN')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Averages_comp/avgcomp_ROC_ann.png')

    # Curva Fscore
    plt.clf()

    for crv, mdl in zip(step5_fscore_curves, best_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    for crv, mdl in zip(avg_das_fscore_curves, avg_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label=mdl.strip().split('.')[0])

    plt.title(f'Fscore vs thresholds curves best models and average models ANN')
    plt.xlabel('Threshold')
    plt.ylabel('F-score')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Averages_comp/avgcomp_Fscore_ann.png')

    # Lo mismo de arriba pero solo dos modelos

    # Curva PR
    plt.clf()

    for crv, mdl in zip(step5_pr_curves, best_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label='best')
        break

    for crv, mdl in zip(avg_das_pr_curves, avg_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label='average')
        break

    # Dumb model line
    plt.hlines(0.5, 0, 1, 'b', '--')
    plt.title(f'PR curve best model and average model ANN')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(-0.02, 1.02)
    plt.ylim(0.48, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Averages_comp/avgcomp_one_PR_ann.png')

    # Curva ROC
    plt.clf()

    for crv, mdl in zip(step5_roc_curves, best_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label='best')
        break

    for crv, mdl in zip(avg_das_roc_curves, avg_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label='average')
        break

    # Dumb model line
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC curve best model and average model ANN')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Averages_comp/avgcomp_one_ROC_ann.png')

    # Curva Fscore
    plt.clf()

    for crv, mdl in zip(step5_fscore_curves, best_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label='best')
        break

    for crv, mdl in zip(avg_das_fscore_curves, avg_models):
        plt.plot(crv[0], crv[1], marker='.', markersize=6, label='average')
        break

    plt.title(f'Fscore vs thresholds curve best model and average model ANN')
    plt.xlabel('Threshold')
    plt.ylabel('F-score')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'../Analysis/Final/Averages_comp/avgcomp_one_Fscore_ann.png')

    # Comparar las curvas PR de STEAD Y DAS mejores modelos individualemente

    for (pr_step4, pr_step5, mdl) in zip(step4_pr_curves, step5_pr_curves, best_models):
        mdl = mdl.split('.')[0].strip()
        plt.clf()

        line_st4, = plt.plot(pr_step4[0], pr_step4[1], marker='.', markersize=6, label="STEAD PR curve")
        line_st5, = plt.plot(pr_step5[0], pr_step5[1], marker='.', markersize=6, label="DAS PR curve")

        plt.hlines(0.5, 0, 1, 'b', '--')
        plt.title(f'Comparacion curva PR STEAD y DAS modelo {mdl}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(-0.02, 1.02)
        plt.ylim(0.48, 1.02)
        plt.grid(True)
        plt.legend(handles=[line_st4, line_st5], loc='best', frameon=False)
        plt.savefig(f'../Analysis/Final/PR_curves/Comparacion_PR_{mdl}.png')

    # Comparar las curvas ROC de STEAD Y DAS mejores modelos individualemente
    for (roc_step4, roc_step5, mdl) in zip(step4_roc_curves, step5_roc_curves, best_models):
        mdl = mdl.split('.')[0].strip()
        plt.clf()

        line_st4, = plt.plot(roc_step4[0], roc_step4[1], marker='.', markersize=6, label="STEAD ROC curve")
        line_st5, = plt.plot(roc_step5[0], roc_step5[1], marker='.', markersize=6, label="DAS ROC curve")

        plt.plot([0, 1], [0, 1], 'b--')
        plt.title(f'Comparacion curva ROC STEAD y DAS modelo {mdl}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Recall')
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        plt.grid(True)
        plt.legend(handles=[line_st4, line_st5], loc='best', frameon=False)
        plt.savefig(f'../Analysis/Final/ROC_curves/Comparacion_ROC_{mdl}.png')

    # Comparar las curvas Fscore de STEAD Y DAS mejores modelos individualemente
    for (fscore_step4, fscore_step5, mdl) in zip(step4_fscore_curves, step5_fscore_curves, best_models):
        mdl = mdl.split('.')[0].strip()
        plt.clf()

        line_st4, = plt.plot(fscore_step4[0], fscore_step4[1], marker='.', markersize=6, label="STEAD Fscore curve")
        line_st5, = plt.plot(fscore_step5[0], fscore_step5[1], marker='.', markersize=6, label="DAS Fscore curve")

        plt.title(f'Comparacion curva Fscore STEAD y DAS modelo {mdl}')
        plt.xlabel('Umbral')
        plt.ylabel('Fscore')
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        plt.grid(True)
        plt.legend(handles=[line_st4, line_st5], loc='best', frameon=False)
        plt.savefig(f'../Analysis/Final/Fscore_curves/Comparacion_Fscore_{mdl}.png')


if __name__ == "__main__":
    main()
