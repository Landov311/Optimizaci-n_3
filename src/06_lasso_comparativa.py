"""Implementacion from scratch de Regresion Lasso (L1) y comparativa con MCO, Ridge y Elastic Net.

Reutiliza exactamente la misma particion train/test (random_state=42),
estandarizacion sin fuga de datos, validacion cruzada de 5 pliegues y
metricas del pipeline principal (`03_pipeline.py`), ademas de validar contra
`sklearn.linear_model.Lasso`.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, train_test_split

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
FIGS_DIR = ROOT_DIR / "figs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import soft_threshold  # noqa: E402

ACENTO = "#005F69"
GRIS = "#6E6E6E"
DORADO = "#C87D20"
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


class LassoScratch:
    """Regresion Lasso (L1) por descenso por coordenadas con umbralizacion suave.

    Minimiza: 0.5 * ||yc - Xc beta||_2^2 + lam * ||beta||_1
    Actualizacion por coordenada: beta_j <- S_lam(x_j^T r_j) / ||x_j||_2^2
    """

    def __init__(self, lam: float, tol: float = 1e-8, max_iter: int = 15000, fit_intercept: bool = True):
        self.lam = lam
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self.loss_path_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, warm_start: np.ndarray | None = None, track_loss: bool = False):
        _, p = X.shape
        if self.fit_intercept:
            Xc = X - X.mean(axis=0)
            yc = y - y.mean()
        else:
            Xc, yc = X, y

        xj_norm2 = np.sum(Xc ** 2, axis=0)
        beta = np.zeros(p) if warm_start is None else warm_start.copy()
        r = yc - Xc @ beta
        loss_path = [] if track_loss else None

        it = 0
        for it in range(1, self.max_iter + 1):
            max_change = 0.0
            for j in range(p):
                if xj_norm2[j] == 0:
                    continue
                r_j = r + Xc[:, j] * beta[j]
                z_j = Xc[:, j] @ r_j
                beta_new = soft_threshold(z_j, self.lam) / xj_norm2[j]
                delta = beta_new - beta[j]
                if delta != 0.0:
                    r = r_j - Xc[:, j] * beta_new
                    beta[j] = beta_new
                    max_change = max(max_change, abs(delta))
            if track_loss:
                loss = 0.5 * np.sum(r ** 2) + self.lam * np.sum(np.abs(beta))
                loss_path.append(float(loss))
            if max_change < self.tol:
                break

        self.coef_ = beta
        self.intercept_ = float(y.mean() - X.mean(axis=0) @ beta) if self.fit_intercept else 0.0
        self.n_iter_ = it
        self.loss_path_ = loss_path
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.intercept_ + X @ self.coef_


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    resid = y_true - y_pred
    mse = float(np.mean(resid ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot)
    return {"R2": r2, "RMSE": rmse, "MSE": mse, "MAE": mae}


def save_fig(fig, stem: str):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS_DIR / f"{stem}.pdf")
    fig.savefig(FIGS_DIR / f"{stem}.png")
    plt.close(fig)


def main():
    np.random.seed(42)

    # 1) Carga de datos y particion identica a 03_pipeline.py
    x_full = pd.read_csv(DATA_DIR / "X_full.csv")
    y = pd.read_csv(DATA_DIR / "y.csv")["log_price"].values
    feat_names = x_full.columns.tolist()
    X = x_full.values
    n, p = X.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_train_s = (X_train - mu) / sigma
    X_test_s = (X_test - mu) / sigma

    # 2) Validacion cruzada 5-fold para Lasso (L1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lambdas_lasso = np.logspace(-1.5, 2.0, 30)

    cv_rows = []
    for lam in lambdas_lasso:
        errs = []
        n_actives = []
        for tr_idx, va_idx in kf.split(X_train_s):
            Xtr, Xva = X_train_s[tr_idx], X_train_s[va_idx]
            ytr, yva = y_train[tr_idx], y_train[va_idx]
            m2, s2 = Xtr.mean(axis=0), Xtr.std(axis=0)
            s2[s2 == 0] = 1.0
            model = LassoScratch(lam=lam, tol=1e-6, max_iter=8000).fit((Xtr - m2) / s2, ytr)
            pred = model.predict((Xva - m2) / s2)
            errs.append(np.mean((yva - pred) ** 2))
            n_actives.append(int(np.sum(np.abs(model.coef_) > 1e-8)))
        cv_rows.append((float(lam), float(np.mean(errs)), float(np.std(errs)), float(np.mean(n_actives))))

    cv_lasso = pd.DataFrame(cv_rows, columns=["lambda", "cv_mse", "cv_std", "avg_active"])
    cv_lasso.to_csv(DATA_DIR / "cv_lasso.csv", index=False)

    best_idx = cv_lasso["cv_mse"].idxmin()
    lam_min = float(cv_lasso.loc[best_idx, "lambda"])
    mse_min = float(cv_lasso.loc[best_idx, "cv_mse"])
    se_min = float(cv_lasso.loc[best_idx, "cv_std"] / np.sqrt(5))
    lam_1se = float(cv_lasso[cv_lasso["cv_mse"] <= mse_min + se_min]["lambda"].max())

    print(f"Lasso CV: lambda_min = {lam_min:.4f} (MSE cv = {mse_min:.4f}) | lambda_1se = {lam_1se:.4f}")

    # 3) Ajuste final en train (lambda_min y lambda_1se) y evaluacion en test
    lasso_min = LassoScratch(lam=lam_min, tol=1e-8, max_iter=20000).fit(X_train_s, y_train, track_loss=True)
    lasso_1se = LassoScratch(lam=lam_1se, tol=1e-8, max_iter=20000).fit(X_train_s, y_train, track_loss=True)

    pred_tr_min = lasso_min.predict(X_train_s)
    pred_te_min = lasso_min.predict(X_test_s)
    pred_tr_1se = lasso_1se.predict(X_train_s)
    pred_te_1se = lasso_1se.predict(X_test_s)

    m_lasso_min_train = metrics(y_train, pred_tr_min)
    m_lasso_min_test = metrics(y_test, pred_te_min)
    m_lasso_1se_train = metrics(y_train, pred_tr_1se)
    m_lasso_1se_test = metrics(y_test, pred_te_1se)

    n_act_min = int(np.sum(np.abs(lasso_min.coef_) > 1e-8))
    n_act_1se = int(np.sum(np.abs(lasso_1se.coef_) > 1e-8))

    # 4) Validacion contra scikit-learn
    # scikit-learn usa (1/(2n))||y - Xb||^2 + alpha||b||_1 => alpha = lam / n_train
    sk_lasso = Lasso(alpha=lam_min / len(y_train), fit_intercept=True, tol=1e-10, max_iter=200000).fit(
        X_train_s, y_train
    )
    diff_sk_coef = float(np.max(np.abs(lasso_min.coef_ - sk_lasso.coef_)))
    diff_sk_int = float(abs(lasso_min.intercept_ - sk_lasso.intercept_))
    m_sk_lasso_test = metrics(y_test, sk_lasso.predict(X_test_s))

    print(f"Validacion vs sklearn.Lasso: max|dcoef| = {diff_sk_coef:.2e}, |dintercept| = {diff_sk_int:.2e}")

    # 5) Camino de regularizacion Lasso
    lam_path = np.logspace(np.log10(lambdas_lasso.max()) + 0.5, -1.8, 60)
    coef_path = []
    n_active_path = []
    warm = np.zeros(p)
    for lam in lam_path:
        m = LassoScratch(lam=lam, tol=1e-6, max_iter=8000).fit(X_train_s, y_train, warm_start=warm)
        warm = m.coef_.copy()
        coef_path.append(warm.copy())
        n_active_path.append(int(np.sum(np.abs(warm) > 1e-8)))
    coef_path = np.array(coef_path)

    # 6) Comparativa contra MCO, Ridge y Elastic Net ya guardados
    with open(DATA_DIR / "results.json", encoding="utf-8") as f:
        R = json.load(f)
    final_fit = np.load(DATA_DIR / "final_fit.npz", allow_pickle=True)

    rows_comp = [
        {
            "Modelo": "MCO (sin regularizar)",
            "Params": "lam=0",
            "Activas": p,
            "R2_train": R["m_ols_train"]["R2"],
            "R2_test": R["m_ols_test"]["R2"],
            "RMSE_test": R["m_ols_test"]["RMSE"],
            "MSE_test": R["m_ols_test"]["MSE"],
            "MAE_test": R["m_ols_test"]["MAE"],
            "Error_tipico_%": (np.exp(R["m_ols_test"]["MAE"]) - 1) * 100,
        },
        {
            "Modelo": "Ridge (L2)",
            "Params": f"lam={R['lam_min']:.2f}",
            "Activas": p,
            "R2_train": R["m_ridge_train"]["R2"],
            "R2_test": R["m_ridge_test"]["R2"],
            "RMSE_test": R["m_ridge_test"]["RMSE"],
            "MSE_test": R["m_ridge_test"]["MSE"],
            "MAE_test": R["m_ridge_test"]["MAE"],
            "Error_tipico_%": (np.exp(R["m_ridge_test"]["MAE"]) - 1) * 100,
        },
        {
            "Modelo": "Elastic Net (L1+L2)",
            "Params": f"lam={R['lam_opt']:.2f}, alpha={R['alpha_opt']}",
            "Activas": R["en_n_active"],
            "R2_train": R["m_en_train"]["R2"],
            "R2_test": R["m_en_test"]["R2"],
            "RMSE_test": R["m_en_test"]["RMSE"],
            "MSE_test": R["m_en_test"]["MSE"],
            "MAE_test": R["m_en_test"]["MAE"],
            "Error_tipico_%": (np.exp(R["m_en_test"]["MAE"]) - 1) * 100,
        },
        {
            "Modelo": "Lasso (L1, lam_min)",
            "Params": f"lam={lam_min:.2f}",
            "Activas": n_act_min,
            "R2_train": m_lasso_min_train["R2"],
            "R2_test": m_lasso_min_test["R2"],
            "RMSE_test": m_lasso_min_test["RMSE"],
            "MSE_test": m_lasso_min_test["MSE"],
            "MAE_test": m_lasso_min_test["MAE"],
            "Error_tipico_%": (np.exp(m_lasso_min_test["MAE"]) - 1) * 100,
        },
        {
            "Modelo": "Lasso (L1, lam_1se)",
            "Params": f"lam={lam_1se:.2f}",
            "Activas": n_act_1se,
            "R2_train": m_lasso_1se_train["R2"],
            "R2_test": m_lasso_1se_test["R2"],
            "RMSE_test": m_lasso_1se_test["RMSE"],
            "MSE_test": m_lasso_1se_test["MSE"],
            "MAE_test": m_lasso_1se_test["MAE"],
            "Error_tipico_%": (np.exp(m_lasso_1se_test["MAE"]) - 1) * 100,
        },
    ]
    comp_df = pd.DataFrame(rows_comp)
    print("\n=== Comparativa en Test (n_train=576, n_test=144, p=34) ===")
    print(comp_df.round(4).to_string(index=False))

    zeros_en = [feat_names[i] for i in range(p) if abs(final_fit["en_coef"][i]) <= 1e-8]
    zeros_lasso_min = [feat_names[i] for i in range(p) if abs(lasso_min.coef_[i]) <= 1e-8]
    zeros_lasso_1se = [feat_names[i] for i in range(p) if abs(lasso_1se.coef_[i]) <= 1e-8]
    print(f"\nVariables en 0 - Elastic Net ({len(zeros_en)}): {zeros_en}")
    print(f"Variables en 0 - Lasso min   ({len(zeros_lasso_min)}): {zeros_lasso_min}")
    print(f"Variables en 0 - Lasso 1se   ({len(zeros_lasso_1se)}): {zeros_lasso_1se}")

    # 7) Guardar artefactos Lasso
    lasso_results = {
        "n": n,
        "p": p,
        "lam_min": lam_min,
        "mse_cv_min": mse_min,
        "lam_1se": lam_1se,
        "n_iter_min": int(lasso_min.n_iter_),
        "n_active_min": n_act_min,
        "n_iter_1se": int(lasso_1se.n_iter_),
        "n_active_1se": n_act_1se,
        "m_lasso_min_train": m_lasso_min_train,
        "m_lasso_min_test": m_lasso_min_test,
        "m_lasso_1se_train": m_lasso_1se_train,
        "m_lasso_1se_test": m_lasso_1se_test,
        "diff_sk_coef": diff_sk_coef,
        "diff_sk_intercept": diff_sk_int,
        "m_sk_lasso_test": m_sk_lasso_test,
        "zeros_min": zeros_lasso_min,
        "zeros_1se": zeros_lasso_1se,
    }
    with open(DATA_DIR / "lasso_results.json", "w", encoding="utf-8") as f:
        json.dump(lasso_results, f, indent=2, ensure_ascii=False)

    np.savez(
        DATA_DIR / "lasso_fit.npz",
        lasso_min_coef=lasso_min.coef_,
        lasso_min_intercept=lasso_min.intercept_,
        lasso_1se_coef=lasso_1se.coef_,
        lasso_1se_intercept=lasso_1se.intercept_,
        pred_te_lasso_min=pred_te_min,
        pred_te_lasso_1se=pred_te_1se,
        lam_path=lam_path,
        coef_path=coef_path,
        n_active_path=np.array(n_active_path),
        feat_names=np.array(feat_names),
        sigma=sigma,
        mu=mu,
    )

    # 8) Figuras comparativas
    # 8a) Curva CV de Lasso + variables activas
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    se = cv_lasso["cv_std"] / np.sqrt(5)
    axes[0].plot(cv_lasso["lambda"], cv_lasso["cv_mse"], color=ACENTO, lw=1.6)
    axes[0].fill_between(cv_lasso["lambda"], cv_lasso["cv_mse"] - se, cv_lasso["cv_mse"] + se, color=ACENTO, alpha=0.18, lw=0)
    axes[0].axvline(lam_min, color="crimson", ls=":", lw=1.3, label=rf"$\lambda_{{\min}}={lam_min:.2f}$")
    axes[0].axvline(lam_1se, color=GRIS, ls=":", lw=1.3, label=rf"$\lambda_{{1se}}={lam_1se:.2f}$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\lambda$ (escala log)")
    axes[0].set_ylabel("MSE promedio (CV, 5 pliegues)")
    axes[0].set_title("Validación cruzada Lasso ($L_1$)")
    axes[0].legend(frameon=False, fontsize=8.5)

    axes[1].plot(lam_path, n_active_path, color=DORADO, lw=1.6, marker="o", ms=2.5)
    axes[1].axvline(lam_min, color="crimson", ls=":", lw=1.3, label=rf"$\lambda_{{\min}}$ ({n_act_min}/34 activas)")
    axes[1].axvline(lam_1se, color=GRIS, ls=":", lw=1.3, label=rf"$\lambda_{{1se}}$ ({n_act_1se}/34 activas)")
    axes[1].set_xscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel(r"$\lambda$ (escala log, decreciente)")
    axes[1].set_ylabel("N.º de variables no nulas")
    axes[1].set_title("Selección de variables (esparsidad $L_1$)")
    axes[1].legend(frameon=False, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, "lasso_cv")

    # 8b) Camino de regularizacion Lasso
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    cmap = plt.get_cmap("tab20")
    top_idx = np.argsort(-np.abs(coef_path[-1]))[:10]
    for k, j in enumerate(top_idx):
        ax.plot(lam_path, coef_path[:, j], color=cmap(k % 20), lw=1.3, label=feat_names[j])
    for j in range(p):
        if j not in top_idx:
            ax.plot(lam_path, coef_path[:, j], color="0.85", lw=0.7, zorder=0)
    ax.axvline(lam_min, color="crimson", ls=":", lw=1.3, label=rf"$\lambda_{{\min}}={lam_min:.2f}$")
    ax.axvline(lam_1se, color=GRIS, ls=":", lw=1.3, label=rf"$\lambda_{{1se}}={lam_1se:.2f}$")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\lambda$ (escala log, decreciente)")
    ax.set_ylabel(r"$\hat\beta_j$ (coeficientes estandarizados)")
    ax.set_title("Camino de regularización Lasso ($L_1$)")
    ax.legend(fontsize=6.3, ncol=2, frameon=False, loc="upper left")
    plt.tight_layout()
    save_fig(fig, "lasso_reg_path")

    # 8c) Predicho vs real: Ridge vs Elastic Net vs Lasso
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), sharex=True, sharey=True)
    y_test_saved = final_fit["y_test"]
    panels = [
        (final_fit["pred_te_r"], "Ridge ($L_2$)", R["m_ridge_test"]["R2"], GRIS),
        (final_fit["pred_te_e"], "Elastic Net ($L_1+L_2$)", R["m_en_test"]["R2"], ACENTO),
        (pred_te_min, "Lasso ($L_1$)", m_lasso_min_test["R2"], DORADO),
    ]
    lims = [min(y_test_saved.min(), pred_te_min.min()) - 0.1, max(y_test_saved.max(), pred_te_min.max()) + 0.1]
    for ax, (pred, name, r2, col) in zip(axes, panels):
        ax.scatter(y_test_saved, pred, s=14, alpha=0.58, color=col, edgecolors="none")
        ax.plot(lims, lims, "--", color="black", lw=0.9, alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("log(precio) real")
        ax.set_title(f"{name}\n($R^2={r2:.4f}$)")
    axes[0].set_ylabel("log(precio) predicho")
    plt.tight_layout()
    save_fig(fig, "lasso_pred_vs_real")

    # 8d) Coeficientes comparados: Ridge vs Elastic Net vs Lasso
    ridge_coef = final_fit["ridge_coef"]
    en_coef = final_fit["en_coef"]
    order = np.argsort(-np.maximum.reduce([np.abs(ridge_coef), np.abs(en_coef), np.abs(lasso_min.coef_)]))[:14]
    labels = [feat_names[i] for i in order]
    y_pos = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    h = 0.26
    ax.barh(y_pos - h, ridge_coef[order], height=h, color=GRIS, label="Ridge ($L_2$)")
    ax.barh(y_pos, en_coef[order], height=h, color=ACENTO, label="Elastic Net ($L_1+L_2$)")
    ax.barh(y_pos + h, lasso_min.coef_[order], height=h, color=DORADO, label=rf"Lasso ($L_1$, $\lambda_{{\min}}$)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(r"Coeficiente estandarizado $\hat\beta_j$")
    ax.set_title("Comparación de coeficientes: Ridge vs Elastic Net vs Lasso")
    ax.legend(frameon=False, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, "lasso_coef_comparacion")

    print("\nArchivos guardados: data/cv_lasso.csv, data/lasso_results.json, data/lasso_fit.npz")
    print("Figuras guardadas en figs/: lasso_cv, lasso_reg_path, lasso_pred_vs_real, lasso_coef_comparacion (.pdf y .png)")


if __name__ == "__main__":
    main()
