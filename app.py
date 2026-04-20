import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np
import itertools
from statsmodels.formula.api import ols
import plotly.express as px
import plotly.graph_objects as go
from math import log10, floor

st.set_page_config(page_title="Analyse Plan d'Expérience", layout="wide")

# --- FONCTIONS DE FORMATAGE DES CHIFFRES SIGNIFICATIFS ---
def sig_figs(x, n=3):
    """Calcule la valeur arrondie à n chiffres significatifs"""
    if x == 0 or not np.isfinite(x):
        return x
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))

def format_val(val, n=3):
    """Transforme une valeur en texte formaté pour l'affichage individuel"""
    if isinstance(val, (int, float, np.float64)):
        if val == 0 or not np.isfinite(val): return "0"
        res = sig_figs(val, n)
        return f"{res:g}" # Format compact (enlève les zéros inutiles)
    return str(val)

def apply_sig_figs_df(df, n=3):
    """Applique le formatage cellule par cellule à un DataFrame pour l'affichage"""
    func = getattr(df, 'map', getattr(df, 'applymap', None))
    return func(lambda x: format_val(x, n))

st.title("📊 Analyse Plan d'Expérience")
st.write("Equation - Graphique de Pareto - Diagramme des effets - Surface de réponse - Variance - Simulateur - Optimisation multi-objectifs et front de Pareto")
st.write("**Par : GRANCHER Arthur - JACQUET-PIERROULET Pierre - PASQUIER Florent - THALMANN Thomas**")

# --- 1. CHARGEMENT ---
st.sidebar.markdown("# ⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Charger le fichier", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
                except:
                    pass

        st.header("Données expérimentales")
        st.dataframe(df.head())

        colonnes = df.columns.tolist()
        facteurs = st.sidebar.multiselect("Entrées (X)", options=colonnes)
        reponses = st.sidebar.multiselect("Sorties (Y)", options=colonnes)
        
        types_facteurs = {}
        if facteurs:
            st.sidebar.markdown("# 📐 Type d'entrée")
            for f in facteurs:
                if df[f].dtype == 'object':
                    st.sidebar.text(f"{f} : Catégoriel")
                    types_facteurs[f] = "Catégoriel"
                else:
                    types_facteurs[f] = st.sidebar.selectbox(f" {f}", ["Continu", "Catégoriel"], key=f"t_{f}")
        
        st.sidebar.markdown("# 📈 Modèle de régression")
        type_modele = st.sidebar.selectbox(
            "Modèle", 
            ["Linéaire (Simple)", "Interactions (A*B)", "Quadratique (RSM)"],
            label_visibility="collapsed"
        )

        if facteurs and reponses:
            models_dict = {}

            for c_idx, cible in enumerate(reponses):
                st.divider()
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 10px solid #1f77b4;">
                    <h1 style="margin: 0; font-size: 30px;">  Analyse de : {cible}</h1>
                </div>
                <br>
                """, unsafe_allow_html=True)

                facteurs_q = [f'C(Q("{f}"))' if types_facteurs[f] == "Catégoriel" else f'Q("{f}")' for f in facteurs]
                cible_q = f'Q("{cible}")'

                if type_modele == "Linéaire (Simple)":
                    formula = f"{cible_q} ~ {' + '.join(facteurs_q)}"
                elif type_modele == "Interactions (A*B)":
                    formula = f"{cible_q} ~ ({' + '.join(facteurs_q)})**2"
                else:
                    terms = []
                    for i, f in enumerate(facteurs):
                        terms.append(facteurs_q[i])
                        if types_facteurs[f] == "Continu":
                            terms.append(f'I(Q("{f}")**2)')
                    formula = f"{cible_q} ~ ({' + '.join(terms)})**2"

                model = ols(formula, data=df).fit()
                models_dict[cible] = model

                st.subheader("Modèle mathématique")
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1: st.metric("R² (Standard)", format_val(model.rsquared, 4))
                with c2: st.metric("R² Ajusté", format_val(model.rsquared_adj, 4))
                with c3:
                    params = model.params
                    eq_parts = [f"({format_val(val, 3)} * {name.replace('Q(','').replace(')','').replace('C(','')})" for name, val in params.items() if name != 'Intercept']
                    st.info(f"📝 **Équation :**\n\n**{cible}** = {' + '.join(eq_parts)} + ({format_val(params['Intercept'], 3)})")

                st.subheader("Graphique de Pareto")
                st.info(f"💡 Le graphique de Pareto affiche l'influence de chaque facteur via le test de Student ($|t|$) : toute barre dépassant la ligne rouge (seuil de 2.0) est statistiquement significative avec un niveau de confiance de 95 %. ")
                t_vals = model.tvalues.drop('Intercept', errors='ignore')
                importance = pd.DataFrame({
                    'Terme': [t.replace('Q("', '').replace('")', '').replace('C(', '').replace(')', '').replace('I(', '').replace('**2', '²') for t in t_vals.index], 
                    'T': [sig_figs(v, 3) for v in np.abs(t_vals.values)]
                }).sort_values('T')
                fig_p = px.bar(importance, x='T', y='Terme', orientation='h', color='T', template="plotly_white")
                fig_p.add_vline(x=2.0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_p, use_container_width=True, key=f"pareto_{cible}")

                st.subheader("Influence des entrées")
                y_min, y_max = df[cible].min(), df[cible].max()
                margin = (y_max - y_min) * 0.05
                range_y = [y_min - margin, y_max + margin]
                cols_eff = st.columns(len(facteurs))
                for f_idx, f in enumerate(facteurs):
                    with cols_eff[f_idx]:
                        means = df.groupby(f)[cible].mean().reset_index()
                        fig_eff = go.Figure(go.Scatter(x=means[f].astype(str), y=means[cible], mode='lines+markers', line=dict(color='#1f77b4', width=4), marker=dict(size=12)))
                        fig_eff.update_layout(title=f"Effet {f}", yaxis=dict(range=range_y), height=300, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white")
                        st.plotly_chart(fig_eff, use_container_width=True, key=f"eff_{cible}_{f}")

                facteurs_continus = [f for f in facteurs if types_facteurs[f] == "Continu"]
                if len(facteurs_continus) >= 2:
                    st.subheader("Surface de Réponse")
                    cx, cy = st.columns(2)
                    with cx: x_axis_3d = st.selectbox(f"Axe X (3D)", facteurs_continus, index=0, key=f"sx3d_{cible}")
                    with cy: y_axis_3d = st.selectbox(f"Axe Y (3D)", facteurs_continus, index=1, key=f"sy3d_{cible}")
                    if x_axis_3d != y_axis_3d:
                        x_r = np.linspace(df[x_axis_3d].min(), df[x_axis_3d].max(), 30)
                        y_r = np.linspace(df[y_axis_3d].min(), df[y_axis_3d].max(), 30)
                        xx, yy = np.meshgrid(x_r, y_r)
                        df_s = pd.DataFrame({x_axis_3d: xx.flatten(), y_axis_3d: yy.flatten()})
                        for f_other in facteurs:
                            if f_other not in [x_axis_3d, y_axis_3d]:
                                df_s[f_other] = df[f_other].iloc[0] if types_facteurs[f_other] == "Catégoriel" else df[f_other].mean()
                        try:
                            zz = model.predict(df_s).values.reshape(xx.shape)
                            fig_3d = go.Figure(data=[go.Surface(z=zz, x=x_r, y=y_r, colorscale='Viridis')])
                            fig_3d.update_layout(scene=dict(xaxis_title=x_axis_3d, yaxis_title=y_axis_3d, zaxis_title=cible), height=500, margin=dict(l=0,r=0,b=0,t=0))
                            st.plotly_chart(fig_3d, use_container_width=True, key=f"3d_{cible}")
                        except Exception as e:
                            st.warning(f"Erreur surface 3D : {e}")
                
                st.subheader("Analyse de la variance - ANOVA")
                res_df = pd.DataFrame({"Coef": model.params, "Std Err": model.bse, "t": model.tvalues, "P>|t|": model.pvalues})
                res_df.index = [i.replace('Q("', '').replace('")', '').replace('C(', '').replace(')', '') for i in res_df.index]
                # Affichage formaté individuellement
                st.dataframe(apply_sig_figs_df(res_df, 3), use_container_width=True)

            # --- 2. SIMULATEUR ---
            st.divider()
            st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 10px solid #1f77b4;">
                    <h1 style="margin: 0; font-size: 30px;">   Simulateur de réponse </h1>
                </div>
                <br>
                """, unsafe_allow_html=True)

            sc1, sc2 = st.columns(2)
            inputs_sim = {}
            with sc1:
                for f in facteurs:
                    if types_facteurs[f] == "Catégoriel": 
                        inputs_sim[f] = st.selectbox(f"{f}", options=df[f].unique(), key=f"s_{f}")
                    else: 
                        f_min, f_max, f_mean = float(df[f].min()), float(df[f].max()), float(df[f].mean())
                        inputs_sim[f] = st.slider(f"{f}", f_min, f_max, f_mean, key=f"s_{f}")

            with sc2:
                for cible, mod in models_dict.items():
                    try:
                        prediction = mod.predict(pd.DataFrame([inputs_sim]))[0]
                        st.metric(f"Prédiction {cible}", format_val(prediction, 4))
                    except:
                        st.write(f"Calcul en attente...")

            # --- 3. OPTIMISATION ---
            st.divider()
            st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 10px solid #1f77b4;">
                    <h1 style="margin: 0; font-size: 30px;">  Optimisation multi-objectifs et front de Pareto </h1>
                </div>
                <br>
                """, unsafe_allow_html=True)
            
            st.subheader("⚙️ Précision de la recherche")
            st.info(f"💡 Plus la précision est haute, plus le nombre de combinaisons testées est important et le temps de calcul long. ")
            niveaux_label = ["Très Rapide", "Rapide", "Standard", "Précis", "Haute Précision"]
            niveaux_valeurs = [5, 10, 15, 25, 40]
            selection_p = st.select_slider("Précision du calcul", options=niveaux_label, value="Standard")
            precision_opti = niveaux_valeurs[niveaux_label.index(selection_p)]
            
            configs = {}
            niveaux_poids = {"Faible": 1, "Moyen": 5, "Élevé": 10}
            objs_cols = st.columns(len(reponses))
            for i, cible in enumerate(reponses):
                with objs_cols[i]:
                    st.markdown(f"**{cible}**")
                    obj = st.selectbox("Objectif", ["Maximiser", "Minimiser", "Cible", "Ignorer"], key=f"o_{cible}")
                    t_val = st.number_input(f"Visée", value=float(df[cible].mean()), key=f"v_{cible}") if obj == "Cible" else 0.0
                    poids = st.select_slider("Poids", options=["Faible", "Moyen", "Élevé"], value="Moyen", key=f"p_{cible}")
                    configs[cible] = {"obj": obj, "poids": niveaux_poids[poids], "target": t_val}

            st.subheader("Minimum/Maximum des entrées")
            bornes_contraintes = {}
            cols_bornes = st.columns(len(facteurs))
            for i, f in enumerate(facteurs):
                with cols_bornes[i]:
                    if types_facteurs[f] == "Continu":
                        f_min, f_max = float(df[f].min()), float(df[f].max())
                        bornes_contraintes[f] = st.slider(f"Bornes {f}", f_min, f_max, (f_min, f_max), key=f"b_{f}")
                    else:
                        bornes_contraintes[f] = df[f].unique()

            st.subheader("Nombre de combinaisons à afficher")
            nb_top = st.number_input("", min_value=1, max_value=500, value=10)

            if st.button("🚀 Calculer les Optimums"):
                with st.spinner("Calcul en cours..."):
                    def calc_desirability(dataframe, configs, reponses, df_ref):
                        d_total = dataframe.copy()
                        d_total['Score_Log'] = 0.0
                        tw = 0.0
                        for cible in reponses:
                            conf = configs[cible]
                            if conf["obj"] != "Ignorer":
                                c_min, c_max = df_ref[cible].min(), df_ref[cible].max()
                                rv = (c_max - c_min) if (c_max - c_min) != 0 else 1.0
                                if conf["obj"] == "Maximiser": d = (d_total[cible] - c_min) / rv
                                elif conf["obj"] == "Minimiser": d = (c_max - d_total[cible]) / rv
                                elif conf["obj"] == "Cible": d = 1 - (np.abs(d_total[cible] - conf["target"]) / rv)
                                d = np.clip(np.array(d, dtype=float), 1e-6, 1.0)
                                d_total['Score_Log'] += conf["poids"] * np.log(d)
                                tw += conf["poids"]
                        d_total['Score_Final'] = np.exp(d_total['Score_Log'] / tw) if tw > 0 else 0
                        return d_total

                    res_real = calc_desirability(df, configs, reponses, df)
                    best_r = res_real.sort_values('Score_Final', ascending=False).iloc[0]

                    grid = {}
                    for f in facteurs:
                        grid[f] = bornes_contraintes[f] if types_facteurs[f] == "Catégoriel" else np.linspace(bornes_contraintes[f][0], bornes_contraintes[f][1], precision_opti)
                    
                    df_grid = pd.DataFrame([dict(zip(grid.keys(), v)) for v in itertools.product(*grid.values())])
                    for cible, mod in models_dict.items():
                        df_grid[cible] = mod.predict(df_grid)
                    
                    res_theo = calc_desirability(df_grid, configs, reponses, df)
                    top_solutions = res_theo.sort_values('Score_Final', ascending=False).head(nb_top)
                    best_t = top_solutions.iloc[0]

                    st.divider()
                    co1, co2 = st.columns(2)
                    with co1:
                        st.success(f"📍 **Meilleur Essai Réalisé** (Score: {format_val(best_r['Score_Final'], 3)})")
                        # Affichage formaté pour le récapitulatif
                        df_best_r = best_r[facteurs + reponses].to_frame().rename(columns={best_r.name: "Valeur"})
                        st.dataframe(apply_sig_figs_df(df_best_r, 4), use_container_width=True)
                    with co2:
                        st.info(f"🔮 **Optimum Théorique** (Score: {format_val(best_t['Score_Final'], 3)})")
                        df_best_t = best_t[facteurs + reponses].to_frame().rename(columns={best_t.name: "Valeur"})
                        st.dataframe(apply_sig_figs_df(df_best_t, 4), use_container_width=True)

                    st.divider()
                    st.subheader(f"{nb_top} meilleures combinaisons")
                    # Affichage du tableau complet formaté
                    st.dataframe(apply_sig_figs_df(top_solutions[facteurs + reponses + ['Score_Final']], 4), use_container_width=True)

                    if len(reponses) >= 2:
                        st.subheader(f"📉 Front de Pareto ({nb_top} combinaisons)")
                        fig_type = px.scatter_3d if len(reponses) >= 3 else px.scatter
                        kwargs = {"z": reponses[2]} if len(reponses) >= 3 else {}
                        fig_pareto = fig_type(top_solutions, x=reponses[0], y=reponses[1], color='Score_Final', hover_data=facteurs, color_continuous_scale='Viridis', **kwargs)
                        st.plotly_chart(fig_pareto, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur globale : {e}")