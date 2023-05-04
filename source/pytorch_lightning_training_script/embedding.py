import json
import traceback
import lineNotifier
from transformers import AutoTokenizer
import os
from pretrain_lstm import Specter
import time
import shutil
import glob

"""
アブストラクトの分類結果を独自データで追加学習したSPECTERを用いて埋め込みをする
"""


def main():
    """
    データセットのパスなどを代入
    """

    size = "medium"

    # 用いる観点をリストで入力
    labelList = ["title", "bg", "obj", "method", "res"]

    # モデルパラメータのパス
    epoch = 0
    files = glob.glob(
        f"../dataserver/model_outputs/specter/20230503/version_0/checkpoints/*")
    # print(files)
    for filePath in files:
        if f"ep-epoch={epoch}" in filePath:
            modelCheckpoint = filePath
            break
    outputName = "lstm"

    # 入力（埋め込む論文アブストラクトデータ）
    dirPath = "../dataserver/axcell/" + size
    dataPath = dirPath + "/paperDict.json"
    labeledAbstPath = dirPath + "/labeledAbst.json"

    # 出力（文埋め込み）
    outputDirPath = dirPath + "-" + outputName + "/"
    outputEmbLabelDirPath = outputDirPath + "embLabel/"
    if not os.path.exists(outputDirPath):
        os.mkdir(outputDirPath)
    if not os.path.exists(outputEmbLabelDirPath):
        os.mkdir(outputEmbLabelDirPath)

    """
    データのロード・整形
    """
    # データセットをロード
    with open(dataPath, 'r') as f:
        paperDict = json.load(f)

    with open(outputDirPath + "paperDict.json", "w") as f:
        json.dump(paperDict, f, indent=4)

    # 分類されたアブストラクトをロード
    with open(labeledAbstPath, 'r') as f:
        labeledAbstDict = json.load(f)

    shutil.copy(labeledAbstPath, outputDirPath)

    # 扱いやすいようにアブストだけでなくタイトルもvalueで参照できるようにしておく
    for title in labeledAbstDict:
        labeledAbstDict[title]["title"] = title

    try:
        """
        観点毎のデータで学習した観点毎のSPECTERモデルで埋め込み
        """
        # 出力用
        labeledAbstSpecter = {}

        # モデルの初期化
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        model = Specter.load_from_checkpoint(modelCheckpoint)
        # model.cuda(1)
        model.cuda()
        model.eval()

        print("--path: {}--".format(modelCheckpoint))

        for i, label in enumerate(labelList):
            print("--label: {}--".format(label))
            count = 0
            # 埋め込み
            for title, paper in paperDict.items():
                if not title in labeledAbstSpecter:
                    labeledAbstSpecter[title] = {}

                if labeledAbstDict[title][label]:
                    input = tokenizer(
                        # 文の間には[SEP]を挿入しない（挿入した方が良かったりする？）
                        labeledAbstDict[title][label],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=512
                        # ).to('cuda:1')
                    ).to('cuda:0')

                    count += 1

                    # print(input)
                    output = model(**input)[0].tolist()
                    # print(output)
                    # print(count, labeledAbstDict[title][label])
                    labeledAbstSpecter[title][label] = output

                    # debug
                    print(output)
                    exit()

                else:
                    labeledAbstSpecter[title][label] = None

        # ファイル出力
        with open(outputEmbLabelDirPath + "labeledAbstSpecter.json", 'w') as f:
            json.dump(labeledAbstSpecter, f, indent=4)

        message = "【完了】shape-and-emmbedding.py"
        lineNotifier.line_notify(message)

    except Exception as e:
        print(traceback.format_exc())
        message = "Error: " + \
            os.path.basename(__file__) + " " + str(traceback.format_exc())
        lineNotifier.line_notify(message)


if __name__ == '__main__':
    main()
