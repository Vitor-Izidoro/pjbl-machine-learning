# 
# 1. IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS
# 
# Importo bibliotecas principais para manipulação de dados e ML
import pandas as pd
import numpy as np
from scipy.io import arff
import warnings

# Modelos de classificação e regressão exigidos pelo PjBL
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)

# Ferramentas de pré-processamento e criação de pipelines
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# Ferramentas para avaliação de modelos (hold-out e k-fold)
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, r2_score, mean_squared_error, mean_absolute_error,
    make_scorer
)

# Ignoro avisos que não interferem nos resultados (pra deixar a saída limpa)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

print("Bibliotecas importadas com sucesso.")

# 
# 2. FUNÇÕES AUXILIARES
# 

def load_arff_to_df(path):
    """Carrega arquivo .arff e converte em DataFrame."""
    # Função compatível com datasets do WEKA/MOA
    try:
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        # Decodifica strings (os ARFF vêm com bytes)
        for col in df.select_dtypes([np.object_]).columns:
            df[col] = df[col].str.decode('utf-8')
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo '{path}' não encontrado.")
        print("Verifique o diretório do script.")
        return None

def specificity_score(y_true, y_pred):
    """Calcula especificidade = TN / (TN + FP)."""
    # sklearn não possui essa métrica, então implementei manualmente
    if len(np.unique(y_true)) > 2:
        # Versão multiclasse — calcula média por classe
        cm = confusion_matrix(y_true, y_pred)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        specificity = np.nan_to_num(tn / (tn + fp))
        return np.mean(specificity)
    else:
        # Versão binária simples
        cm_flat = confusion_matrix(y_true, y_pred).ravel()
        if len(cm_flat) == 4:
            tn, fp, fn, tp = cm_flat
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            return 0.0

def get_classification_metrics(y_true, y_pred):
    """Calcula as métricas da parte de classificação."""
    # Retorna todas as colunas da tabela A e B
    return {
        'Taxa de Acerto (%)': accuracy_score(y_true, y_pred) * 100,
        'F1 (%)': f1_score(y_true, y_pred, average='macro') * 100,
        'Precisão (%)': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'Sensibilidade (%)': recall_score(y_true, y_pred, average='macro') * 100,
        'Especificidade (%)': specificity_score(y_true, y_pred) * 100
    }

def get_regression_metrics(y_true, y_pred):
    """Calcula as métricas da parte de regressão."""
    # metrica pedidas nas Tabelas C e D
    return {
        'Coeficiente de Determinação (R2)': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

print("Funções auxiliares definidas.")

# 
# 3. PREPARAÇÃO DOS DADOS
# 

# Carrega os datasets .arff
df_class = load_arff_to_df('powersupply.arff')
df_reg = load_arff_to_df('cpu.arff')

# --- Pré-processamento da Classificação ---
if df_class is not None:
    print(f"\nDataset de Classificação (PowerSupply) carregado: {df_class.shape[0]} linhas, {df_class.shape[1]} colunas")
    # Separo as features e o alvo
    X_class = df_class.drop('class', axis=1).values
    y_class_str = df_class['class'].values
    # Transformo as classes textuais em números
    le = LabelEncoder()
    y_class = le.fit_transform(y_class_str)
    # Normaliza os dados né
    preprocessor_class = StandardScaler()

# --- Pré-processamento da Regressão ---
if df_reg is not None:
    print(f"Dataset de Regressão (CPU) carregado: {df_reg.shape[0]} linhas, {df_reg.shape[1]} colunas")
    target_reg = 'class'  # nome da variável dependente padrão em ARFF
    
    if target_reg in df_reg.columns:
        # Separo entradas e saídas
        X_reg = df_reg.drop(columns=[target_reg])
        y_reg = df_reg[target_reg].values
        preprocessor_reg = StandardScaler()  # normalização simples
        print("Pré-processamento configurado para ambos os datasets.")
    else:
        print(f"\nERRO: O arquivo 'cpu.arff' não contém a coluna-alvo '{target_reg}'.")
        df_reg = None  # Evita erro adiante

# 
# 4. DEFINIÇÃO DOS MODELOS
# 

# Aqui vou definir todos os algoritmos exigidos nas tabela

# --- Modelos de Classificação ---
estimators_class = [('dt', DecisionTreeClassifier(random_state=42)), ('knn', KNeighborsClassifier())]
voting_estimators_class = [('rf', RandomForestClassifier(random_state=42)), ('svm', SVC(probability=True, random_state=42)), ('nb', GaussianNB())]

models_class = {
    "KNN": KNeighborsClassifier(), 
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(), 
    "MLP": MLPClassifier(max_iter=500, random_state=42),
    "SVM": SVC(random_state=42),
    "Ensemble (somatória)": VotingClassifier(estimators=voting_estimators_class, voting='soft'),
    "Random Forest": RandomForestClassifier(random_state=42), 
    "Bagging": BaggingClassifier(random_state=42),
    "Boosting": AdaBoostClassifier(random_state=42),
    "Stacking": StackingClassifier(estimators=estimators_class, final_estimator=RandomForestClassifier(random_state=42)),
    "Blending": StackingClassifier(estimators=estimators_class, final_estimator=RandomForestClassifier(random_state=42), cv=2),
    "Adicional": GradientBoostingClassifier(random_state=42)
}

# --- Modelos de Regressão ---
estimators_reg = [('dt', DecisionTreeRegressor(random_state=42)), ('knn', KNeighborsRegressor())]
voting_estimators_reg = [('rf', RandomForestRegressor(random_state=42)), ('svr', SVR()), ('lr', LinearRegression())]

models_reg = {
    "Regressão Linear": LinearRegression(), 
    "KNN": KNeighborsRegressor(),
    "Árvore de Decisão": DecisionTreeRegressor(random_state=42), 
    "MLP": MLPRegressor(max_iter=1000, random_state=42),
    "SVM": SVR(), 
    "Ensemble (Média)": VotingRegressor(estimators=voting_estimators_reg),
    "Random Forest": RandomForestRegressor(random_state=42), 
    "Bagging": BaggingRegressor(random_state=42),
    "Boosting": AdaBoostRegressor(random_state=42),
    "Stacking": StackingRegressor(estimators=estimators_reg, final_estimator=RandomForestRegressor(random_state=42)),
    "Blending": StackingRegressor(estimators=estimators_reg, final_estimator=RandomForestRegressor(random_state=42), cv=2),
    "Adicional": GradientBoostingRegressor(random_state=42)
}
print("Modelos definidos.")

# 
# 5. EXECUÇÃO DOS EXPERIMENTOS
# 

# Aqui são geradas as 4 tables (A, B, C, D)
# Cada bloco executa todos os modelos e armazena as métricas

if df_class is not None:
    # --- Tabela A: Classificação com Hold-out ---
    print("\n--- Processando Tabela A: Classificação com Hold-out (65/35) ---")
    results_A = []
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.35, random_state=42, stratify=y_class)
    for name, model in models_class.items():
        # Uso de pipeline pra aplicar normalização automaticamente
        pipeline = Pipeline(steps=[('preprocessor', preprocessor_class), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        # Calculo as métricas pedidas no PjBL
        metrics = get_classification_metrics(y_test, y_pred)
        metrics['Indutor'] = name
        results_A.append(metrics)
        print(f"  - {name}: Concluído")  # comentário rápido: indica progresso
    df_results_A = pd.DataFrame(results_A)[['Indutor', 'Taxa de Acerto (%)', 'F1 (%)', 'Precisão (%)', 'Sensibilidade (%)', 'Especificidade (%)']]

    # --- Tabela B: Classificação com Validação Cruzada (5 folds) ---
    print("\n--- Processando Tabela B: Classificação com Validação Cruzada (5 folds) ---")
    results_B = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring_class = {
        'accuracy': 'accuracy', 
        'f1_macro': 'f1_macro', 
        'precision_macro': 'precision_macro', 
        'recall_macro': 'recall_macro', 
        'specificity': make_scorer(specificity_score)
    }
    for name, model in models_class.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor_class), ('classifier', model)])
        scores = cross_validate(pipeline, X_class, y_class, cv=kfold, scoring=scoring_class)
        results_B.append({
            'Indutor': name,
            'Taxa de Acerto (%)': scores['test_accuracy'].mean() * 100,
            'F1 (%)': scores['test_f1_macro'].mean() * 100,
            'Precisão (%)': scores['test_precision_macro'].mean() * 100,
            'Sensibilidade (%)': scores['test_recall_macro'].mean() * 100,
            'Especificidade (%)': scores['test_specificity'].mean() * 100
        })
        print(f"  - {name}: Concluído")
    df_results_B = pd.DataFrame(results_B)[['Indutor', 'Taxa de Acerto (%)', 'F1 (%)', 'Precisão (%)', 'Sensibilidade (%)', 'Especificidade (%)']]

if df_reg is not None:
    # --- Tabela C: Regressão com Hold-out ---
    print("\n--- Processando Tabela C: Regressão com Hold-out (65/35) ---")
    results_C = []
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.35, random_state=42)
    for name, model in models_reg.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor_reg), ('regressor', model)])
        pipeline.fit(X_train_r, y_train_r)
        y_pred_r = pipeline.predict(X_test_r)
        metrics = get_regression_metrics(y_test_r, y_pred_r)
        metrics['Indutor'] = name
        results_C.append(metrics)
        print(f"  - {name}: Concluído")
    df_results_C = pd.DataFrame(results_C)[['Indutor', 'Coeficiente de Determinação (R2)', 'MSE', 'RMSE', 'MAE']]

    # --- Tabela D: Regressão com Validação Cruzada (5 folds) ---
    print("\n--- Processando Tabela D: Regressão com Validação Cruzada (5 folds) ---")
    results_D = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring_reg = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    for name, model in models_reg.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor_reg), ('regressor', model)])
        scores = cross_validate(pipeline, X_reg, y_reg, cv=kfold, scoring=scoring_reg)
        results_D.append({
            'Indutor': name,
            'Coeficiente de Determinação (R2)': scores['test_r2'].mean(),
            'MSE': -scores['test_neg_mean_squared_error'].mean(),
            'RMSE': np.sqrt(-scores['test_neg_mean_squared_error'].mean()),
            'MAE': -scores['test_neg_mean_absolute_error'].mean()
        })
        print(f"  - {name}: Concluído")
    df_results_D = pd.DataFrame(results_D)[['Indutor', 'Coeficiente de Determinação (R2)', 'MSE', 'RMSE', 'MAE']]

# 
# 6. EXIBIÇÃO DOS RESULTADOS
# 

# Exiba todas as tabelas no formato exigido pelo enunciado
# Comentário rápido: essas saídas correspondem às quatro tabelas q tão no PDF

pd.set_option('display.float_format', lambda x: '%.3f' % x)
print("\n\n" + "="*80)
print("RESULTADOS FINAIS")
print("="*80)

if 'df_results_A' in locals():
    print("\n\nTABELA A: Classificação com protocolo Hold-out (65% para treinamento e 35% para teste)")
    print(df_results_A.to_string(index=False))
    print("\n\nTABELA B: Classificação com protocolo experimental validação cruzada com 5 folds")
    print(df_results_B.to_string(index=False))

if 'df_results_C' in locals():
    print("\n\nTABELA C: Regressão com protocolo Hold-out (65% para treinamento e 35% para teste)")
    print(df_results_C.to_string(index=False))
    print("\n\nTABELA D: Regressão com protocolo experimental validação cruzada com 5 folds")
    print(df_results_D.to_string(index=False))
