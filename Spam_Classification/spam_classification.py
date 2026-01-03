# 1. 导入依赖库
import jieba
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# 2. 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 3. 数据预处理函数（修复停用词读取逻辑）
def preprocess_text(text):
    """中文文本预处理：分词+停用词过滤"""
    # 读取停用词表（用open避免pandas解析错误）
    stop_words = set()
    try:
        with open("hit_stopwords.txt", "r", encoding="utf-8-sig") as f:
            for line in f.readlines():
                word = line.strip()
                if word:
                    stop_words.add(word)
    except FileNotFoundError:
        print("警告：未找到hit_stopwords.txt，跳过停用词过滤！")
    
    # jieba分词+过滤
    words = jieba.cut(text, cut_all=False)
    filtered_words = []
    for word in words:
        if (word not in stop_words) and (len(word) >= 2) and (not word.isdigit()) and (word.isalnum()):
            filtered_words.append(word.lower())
    return " ".join(filtered_words)

# 4. 数据加载与预处理（修复稀疏矩阵len()报错）
def load_and_preprocess_data(data_path="sogou_spam.csv"):
    """加载数据集+预处理+TF-IDF+划分训练/测试集"""
    # 读取数据
    data = pd.read_csv(data_path, encoding="utf-8")
    print(f"数据集总样本数：{len(data)}，垃圾邮件数：{len(data[data['label']==1])}，正常邮件数：{len(data[data['label']==0])}")
    
    # 预处理
    data["processed_text"] = data["text"].apply(preprocess_text)
    data = data[data["processed_text"] != ""]  # 过滤空文本
    print(f"预处理后有效样本数：{len(data)}")
    
    # TF-IDF特征提取
    tfidf = TfidfVectorizer(
        max_features=None,
        ngram_range=(1, 1),
        min_df=1,
        stop_words=None
    )
    X = tfidf.fit_transform(data["processed_text"])
    y = data["label"]
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 修复：稀疏矩阵用shape[0]获取样本数
    print(f"训练集样本数：{X_train.shape[0]}，测试集样本数：{X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, tfidf

# 5. 模型训练与评估（保留分类数量统计）
def train_and_evaluate(model, model_name, param_desc, X_train, X_test, y_train, y_test):
    """
    训练模型+评估+统计分类数量
    返回：评估结果字典、预测值、参数描述
    """
    # 训练计时
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start_time, 3)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 核心评估指标
    metrics = {
        "模型": model_name,
        "参数": param_desc,
        "准确率": round(accuracy_score(y_test, y_pred), 4),
        "精确率": round(precision_score(y_test, y_pred), 4),
        "召回率": round(recall_score(y_test, y_pred), 4),
        "F1分数": round(f1_score(y_test, y_pred), 4),
        "训练时间(s)": train_time,
        "参数组合": param_desc
    }
    
    # 统计测试集真实/预测的分类数量（0=非垃圾，1=垃圾）
    true_non_spam = len(y_test[y_test == 0])  # 真实非垃圾邮件数
    true_spam = len(y_test[y_test == 1])      # 真实垃圾邮件数
    pred_non_spam = len(y_pred[y_pred == 0])  # 预测非垃圾邮件数
    pred_spam = len(y_pred[y_pred == 1])      # 预测垃圾邮件数
    
    # 打印分类数量
    print(f"\n{model_name} - {param_desc} 分类统计：")
    print(f"测试集真实：非垃圾邮件{true_non_spam}个，垃圾邮件{true_spam}个")
    print(f"模型预测：非垃圾邮件{pred_non_spam}个，垃圾邮件{pred_spam}个")
    print(f"评估指标：准确率={metrics['准确率']}，精确率={metrics['精确率']}，F1={metrics['F1分数']}")
    
    return metrics, y_pred, param_desc

# 6. 需求1：朴素贝叶斯精确率-扩展参数梯度变化图
def plot_nb_precision_curve(nb_results):
    """绘制朴素贝叶斯alpha参数（0.01~10）vs 精确率折线图"""
    # 提取alpha和精确率（按参数顺序排序）
    alpha_order = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    alpha_list = []
    precision_list = []
    for alpha in alpha_order:
        for res in nb_results:
            if f"alpha={alpha}" == res["参数"]:
                alpha_list.append(alpha)
                precision_list.append(res["精确率"])
                break
    
    # 绘制折线图（标注最优参数）
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_list, precision_list, marker="o", linewidth=2.5, markersize=8, color="#1f77b4")
    # 标注每个点的数值
    for a, p in zip(alpha_list, precision_list):
        plt.text(a, p+0.001, f"{p:.4f}", ha="center", fontsize=10)
    # 标注最优参数
    max_idx = np.argmax(precision_list)
    best_alpha = alpha_list[max_idx]
    best_precision = precision_list[max_idx]
    plt.scatter(best_alpha, best_precision, color="red", s=100, zorder=5, label=f"最优参数：alpha={best_alpha}（精确率={best_precision:.4f}）")
    
    # 图表样式
    plt.title("朴素贝叶斯 精确率 vs alpha参数（0.01~10）", fontsize=14, fontweight="bold")
    plt.xlabel("alpha参数", fontsize=12)
    plt.ylabel("精确率", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    plt.xscale("log")  # 对数轴更直观展示参数梯度
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("朴素贝叶斯精确率参数图_扩展梯度.png", dpi=300, bbox_inches="tight")
    print("朴素贝叶斯精确率参数图已保存：朴素贝叶斯精确率参数图_扩展梯度.png")
    plt.close()

# 7. 需求2：随机森林单图展示双参数组合的精确率（热力图）
def plot_rf_precision_heatmap(rf_results):
    """单图展示随机森林n_estimators+max_depth组合的精确率（热力图）"""
    # 提取参数和精确率
    rf_data = []
    for res in rf_results:
        param = res["参数"]
        n_est = int(param.split(",")[0].split("=")[1])
        max_depth = param.split("max_depth=")[1]
        # 统一max_depth格式（None转为"无限制"，便于显示）
        max_depth_label = "无限制" if max_depth == "None" else int(max_depth)
        rf_data.append({
            "n_estimators": n_est,
            "max_depth": max_depth_label,
            "精确率": res["精确率"]
        })
    rf_df = pd.DataFrame(rf_data)
    
    # 构建透视表（n_estimators为行，max_depth为列）
    pivot_df = rf_df.pivot(index="n_estimators", columns="max_depth", values="精确率")
    # 找到最优参数组合
    max_precision = pivot_df.max().max()
    best_params = np.where(pivot_df == max_precision)
    best_n_est = pivot_df.index[best_params[0][0]]
    best_depth = pivot_df.columns[best_params[1][0]]
    
    # 绘制热力图
    plt.figure(figsize=(10, 7))
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={"label": "精确率"})
    # 标注最优参数组合
    plt.text(
        best_params[1][0]+0.5, best_params[0][0]+0.5, 
        f"最优\n{best_n_est}/{best_depth}", 
        ha="center", va="center", color="red", fontsize=12, fontweight="bold"
    )
    plt.title(f"随机森林 精确率 vs n_estimators+max_depth组合（最优：n_est={best_n_est}, depth={best_depth}）", fontsize=14, fontweight="bold")
    plt.xlabel("max_depth（决策树最大深度）", fontsize=12)
    plt.ylabel("n_estimators（决策树数量）", fontsize=12)
    plt.tight_layout()
    plt.savefig("随机森林精确率双参数热力图.png", dpi=300, bbox_inches="tight")
    print("随机森林双参数精确率热力图已保存：随机森林精确率双参数热力图.png")
    plt.close()

# 8. 需求3：SVM（仅线性核）精确率-C值变化图
def plot_svm_linear_precision_curve(svm_results):
    """绘制SVM（线性核）C值（0.01~100）vs 精确率折线图"""
    # 提取C值和精确率（按参数顺序排序）
    c_order = [0.01, 0.1, 1, 10, 100]
    c_list = []
    precision_list = []
    for c in c_order:
        for res in svm_results:
            if f"C={c}, kernel=linear" == res["参数"]:
                c_list.append(c)
                precision_list.append(res["精确率"])
                break
    
    # 绘制折线图（标注最优参数）
    plt.figure(figsize=(10, 6))
    plt.plot(c_list, precision_list, marker="o", linewidth=2.5, markersize=8, color="#ff7f0e")
    # 标注每个点的数值
    for c, p in zip(c_list, precision_list):
        plt.text(c, p+0.001, f"{p:.4f}", ha="center", fontsize=10)
    # 标注最优参数
    max_idx = np.argmax(precision_list)
    best_c = c_list[max_idx]
    best_precision = precision_list[max_idx]
    plt.scatter(best_c, best_precision, color="red", s=100, zorder=5, label=f"最优参数：C={best_c}（精确率={best_precision:.4f}）")
    
    # 图表样式
    plt.title("SVM（线性核）精确率 vs C参数（0.01~100）", fontsize=14, fontweight="bold")
    plt.xlabel("C参数（正则化强度）", fontsize=12)
    plt.ylabel("精确率", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    plt.xscale("log")  # 对数轴展示C值梯度
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("SVM线性核精确率参数图.png", dpi=300, bbox_inches="tight")
    print("SVM线性核精确率参数图已保存：SVM线性核精确率参数图.png")
    plt.close()

# 9. 修复原精度指标对比图（纵轴自适应）
def visualize_results(all_results, X_test, y_test, y_pred_dict):
    """可视化：精度对比+效率对比+混淆矩阵"""
    # 筛选最优结果
    best_results = all_results.loc[all_results.groupby('模型')['F1分数'].idxmax()]
    model_names = best_results["模型"].tolist()
    print(f"\n各模型最优参数组合：")
    print(best_results[["模型", "参数", "准确率", "F1分数"]].to_string(index=False))
    
    # 精度指标对比图（纵轴自适应）
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.25
    accuracy = best_results["准确率"].tolist()
    precision = best_results["精确率"].tolist()
    f1 = best_results["F1分数"].tolist()
    
    # 绘制柱状图
    plt.bar(x - width, accuracy, width, label='准确率', color='#1f77b4')
    plt.bar(x, precision, width, label='精确率', color='#ff7f0e')
    plt.bar(x + width, f1, width, label='F1分数', color='#2ca02c')
    
    # 纵轴自适应
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('指标值（0~1）', fontsize=12)
    plt.title('三种模型最优参数组合的精度指标对比', fontsize=14, fontweight='bold')
    plt.xticks(x, model_names, fontsize=11)
    plt.ylim(min(min(accuracy), min(precision), min(f1)) - 0.01, 1.0)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    # 标注数值
    for i, v in enumerate(accuracy):
        plt.text(i - width, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)
    for i, v in enumerate(precision):
        plt.text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)
    for i, v in enumerate(f1):
        plt.text(i + width, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('精度指标对比图_修复版.png', dpi=300, bbox_inches='tight')
    print("修复版精度指标对比图已保存：精度指标对比图_修复版.png")
    plt.close()
    
    # 效率对比图
    plt.figure(figsize=(8, 6))
    train_time = best_results["训练时间(s)"].tolist()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = plt.bar(model_names, train_time, color=colors)
    for bar, time_val in zip(bars, train_time):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{time_val:.3f}s', ha='center', fontsize=11)
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('训练时间（s）', fontsize=12)
    plt.title('三种模型最优参数组合的效率对比', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('效率对比图.png', dpi=300, bbox_inches='tight')
    print("效率对比图已保存：效率对比图.png")
    plt.close()
    
    # 混淆矩阵
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('三种模型最优参数组合的混淆矩阵', fontsize=16, fontweight='bold')
    labels = ['非垃圾邮件', '垃圾邮件']
    for idx, (model_name, y_pred) in enumerate(y_pred_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=labels, yticklabels=labels, cbar=False)
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('预测标签', fontsize=11)
        axes[idx].set_ylabel('真实标签', fontsize=11)
    plt.tight_layout()
    plt.savefig('混淆矩阵热力图.png', dpi=300, bbox_inches='tight')
    print("混淆矩阵热力图已保存：混淆矩阵热力图.png")
    plt.close()

# 10. 主实验流程（整合所有新需求）
def run_all_experiments(X_train, X_test, y_train, y_test):
    """执行所有模型实验+收集结果"""
    all_results = []
    y_pred_dict = {}  # 最优模型预测值
    nb_results = []   # 朴素贝叶斯所有结果（用于绘图）
    svm_results = []  # SVM所有结果（用于绘图）
    rf_results = []   # 随机森林所有结果（用于绘图）
    
    # 1. 朴素贝叶斯实验（需求1：扩展参数梯度）
    print("\n" + "="*50 + " 朴素贝叶斯实验 " + "="*50)
    alpha_list = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # 新参数梯度
    nb_best_f1 = -1
    nb_best_y_pred = None
    for alpha in alpha_list:
        model = MultinomialNB(alpha=alpha)
        param_desc = f"alpha={alpha}"
        result, y_pred, _ = train_and_evaluate(model, "朴素贝叶斯", param_desc, X_train, X_test, y_train, y_test)
        all_results.append(result)
        nb_results.append(result)
        if result["F1分数"] > nb_best_f1:
            nb_best_f1 = result["F1分数"]
            nb_best_y_pred = y_pred
    y_pred_dict["朴素贝叶斯"] = nb_best_y_pred
    
    # 2. SVM实验（需求3：仅线性核，C值0.01~100）
    print("\n" + "="*50 + " SVM实验（仅线性核） " + "="*50)
    svm_best_f1 = -1
    svm_best_y_pred = None
    # 仅保留线性核，C值取[0.01, 0.1, 1, 10, 100]
    svm_linear_params = [0.01, 0.1, 1, 10, 100]
    for C in svm_linear_params:
        model = SVC(C=C, kernel="linear", random_state=42)
        param_desc = f"C={C}, kernel=linear"
        result, y_pred, _ = train_and_evaluate(model, "SVM", param_desc, X_train, X_test, y_train, y_test)
        all_results.append(result)
        svm_results.append(result)
        if result["F1分数"] > svm_best_f1:
            svm_best_f1 = result["F1分数"]
            svm_best_y_pred = y_pred
    y_pred_dict["SVM"] = svm_best_y_pred
    
    # 3. 随机森林实验（参数不变，仅绘图方式修改）
    print("\n" + "="*50 + " 随机森林实验 " + "="*50)
    rf_best_f1 = -1
    rf_best_y_pred = None
    n_estimators_list = [50, 100, 200]
    max_depth_list = [None, 10, 20]
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            param_desc = f"n_estimators={n_estimators}, max_depth={max_depth}"
            result, y_pred, _ = train_and_evaluate(model, "随机森林", param_desc, X_train, X_test, y_train, y_test)
            all_results.append(result)
            rf_results.append(result)
            if result["F1分数"] > rf_best_f1:
                rf_best_f1 = result["F1分数"]
                rf_best_y_pred = y_pred
    y_pred_dict["随机森林"] = rf_best_y_pred
    
    # 整理结果
    results_df = pd.DataFrame(all_results)
    print("\n" + "="*50 + " 所有实验结果汇总 " + "="*50)
    print(results_df[["模型", "参数", "准确率", "精确率", "F1分数"]].to_string(index=False))
    results_df.to_csv("实验结果汇总.csv", index=False, encoding="utf-8-sig")
    print("\n实验结果已保存：实验结果汇总.csv")
    
    # 执行所有绘图需求
    print("\n" + "="*50 + " 生成所有图表 " + "="*50)
    # 需求1：朴素贝叶斯扩展参数梯度图
    plot_nb_precision_curve(nb_results)
    # 需求3：SVM线性核C值变化图
    plot_svm_linear_precision_curve(svm_results)
    # 需求2：随机森林双参数组合热力图
    plot_rf_precision_heatmap(rf_results)
    # 修复版精度对比图
    visualize_results(results_df, X_test, y_test, y_pred_dict)
    
    return results_df, y_pred_dict

# 11. 主函数（程序入口）
if __name__ == "__main__":
    print("="*50 + " 数据预处理 " + "="*50)
    # 加载数据（确保sogou_spam.csv在同级目录）
    X_train, X_test, y_train, y_test, tfidf = load_and_preprocess_data(data_path="sogou_spam.csv")
    # 执行实验+绘图
    run_all_experiments(X_train, X_test, y_train, y_test)
    print("\n所有实验和绘图完成！生成的文件：")
    print("1. 实验结果汇总.csv")
    print("2. 朴素贝叶斯精确率参数图_扩展梯度.png（0.01~10参数）")
    print("3. SVM线性核精确率参数图.png（0.01~100 C值）")
    print("4. 随机森林精确率双参数热力图.png（单图展示双参数组合）")
    print("5. 精度指标对比图_修复版.png")
    print("6. 效率对比图.png + 混淆矩阵热力图.png")