from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from data_simulator import make_demo_data
from model_utils import prepare_sequences
from sklearn.metrics import confusion_matrix, roc_curve, auc

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "lightweight_lstm.keras"
SCALER_PATH = ROOT / "models" / "scaler.joblib"
META_PATH = ROOT / "models" / "metadata.json"
RESULTS_PATH = ROOT / "results" / "model_comparison.csv"

def build_pdf_report(out_df, threshold):
    """Create a compact downloadable PDF summary from the current predictions."""
    from io import BytesIO
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=14*mm, leftMargin=14*mm,
                            topMargin=12*mm, bottomMargin=12*mm)
    styles = getSampleStyleSheet()
    total = len(out_df)
    risky = int((out_df["Predicted_Category"] == "At-Risk").sum())
    story = [Paragraph("Early At-Risk Student Prediction — Report", styles["Title"]),
             Paragraph("Generated from the current uploaded dataset and trained model.", styles["Normal"]),
             Spacer(1, 8)]
    summary = [["Students analysed", "Flagged At-Risk", "Classified Safe", "Threshold"],
               [str(total), str(risky), str(total-risky), f"{threshold:.0%}"]]
    table = Table(summary, colWidths=[45*mm]*4)
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#173B57")),
                               ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                               ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                               ("GRID", (0,0), (-1,-1), .4, colors.grey),
                               ("PADDING", (0,0), (-1,-1), 7)]))
    story += [table, Spacer(1, 10), Paragraph("Highest predicted risk probabilities (up to 25 records)", styles["Heading2"])]
    id_col = next((c for c in ["Student_ID", "student_id", "ID", "id"] if c in out_df.columns), None)
    cols = ([id_col] if id_col else []) + ["Predicted_Category", "At_Risk_Probability", "Suggested_Action"]
    view = out_df.sort_values("At_Risk_Probability", ascending=False)[cols].head(25).copy()
    view["At_Risk_Probability"] = view["At_Risk_Probability"].map(lambda x: f"{float(x):.1%}")
    rows = [[str(c).replace("_", " ") for c in cols]] + view.fillna("").astype(str).values.tolist()
    report_table = Table(rows, repeatRows=1, colWidths=None)
    report_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DDEBF2")),
                                      ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                                      ("FONTSIZE", (0,0), (-1,-1), 7), ("GRID", (0,0), (-1,-1), .3, colors.grey),
                                      ("VALIGN", (0,0), (-1,-1), "TOP"), ("PADDING", (0,0), (-1,-1), 4)]))
    story += [report_table, Spacer(1, 8), Paragraph("Important: This is a decision-support signal, not a diagnosis or final academic decision. Validate on institution-approved real data before research claims or operational use. Synthetic demo data must not be reported as real-world evidence.", styles["Italic"])]
    doc.build(story)
    return buffer.getvalue()

st.set_page_config(page_title="Early At-Risk Student Prediction", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {padding: 1.25rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#12304a,#236b75); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin:0; font-size:2rem;}
.hero p {color:#e7f4f6; margin:.4rem 0 0 0;}
div[data-testid="stMetric"] {background: rgba(120,140,160,.09); border: 1px solid rgba(120,140,160,.18); padding: 12px 14px; border-radius: 12px;}
.small-note {font-size:.88rem; color: #667085;}
</style>
<div class="hero"><h1>🎓 Early At-Risk Student Prediction</h1><p>Early-warning analytics • Lightweight LSTM • Faculty decision-support dashboard</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Dashboard controls")
    threshold = st.slider("At-risk probability threshold", 0.10, 0.90, 0.50, 0.05,
                          help="Students at or above this probability are flagged At-Risk. Choose a threshold with institutional staff; it is not a universal standard.")
    st.divider()
    st.subheader("About this prototype")
    st.write("Uses weekly early-semester engagement features to estimate a risk signal.")
    st.warning("Predictions support staff review; they must not be used as the sole basis for academic decisions.")
    st.info("The included demonstration records are synthetic, not real student records.")

# Persist loaded data between tabs
if "prediction_df" not in st.session_state:
    st.session_state["prediction_df"] = None
if "prediction_output" not in st.session_state:
    st.session_state["prediction_output"] = None

# Load a sample quickly, while allowing users to upload their own data
with st.expander("🚀 Quick start: load demonstration data", expanded=st.session_state["prediction_df"] is None):
    q1, q2 = st.columns([2, 1])
    q1.write("Load a small sample to explore the dashboard. To generate model predictions, train the model first in **Train & Evaluate**.")
    if q2.button("Load 30 demo students", use_container_width=True):
        st.session_state["prediction_df"] = make_demo_data(n=30, seed=17).drop(columns=["At_Risk"], errors="ignore")
        st.session_state["prediction_output"] = None
        st.rerun()

uploaded = st.file_uploader("Or upload a student CSV", type=["csv"], key="main_upload", help="For prediction, include the same weekly feature columns used during training.")
if uploaded is not None:
    try:
        new_df = pd.read_csv(uploaded)
        st.session_state["prediction_df"] = new_df
        st.session_state["prediction_output"] = None
    except Exception as e:
        st.error(f"Unable to read this CSV: {e}")

df = st.session_state["prediction_df"]
tabs = st.tabs(["📊 Overview", "🔍 Predict & Review", "📈 Student Insights", "🧠 Train & Evaluate", "🧪 Evaluation & Explainability", "📄 Data Guide"])

with tabs[0]:
    st.subheader("At-a-glance overview")
    model_ready = MODEL_PATH.exists() and SCALER_PATH.exists()
    total = len(df) if isinstance(df, pd.DataFrame) else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records loaded", f"{total:,}")
    c2.metric("Model status", "Ready" if model_ready else "Not trained")
    c3.metric("Observation window", "6 weeks")
    c4.metric("Decision threshold", f"{threshold:.0%}")
    out_overview = st.session_state.get("prediction_output")
    if isinstance(out_overview, pd.DataFrame) and len(out_overview):
        risky_n = int((out_overview["Predicted_Category"] == "At-Risk").sum())
        avg_prob = float(out_overview["At_Risk_Probability"].mean())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Predicted At-Risk", f"{risky_n:,}")
        k2.metric("Predicted Safe", f"{len(out_overview)-risky_n:,}")
        k3.metric("At-Risk share", f"{risky_n/len(out_overview):.1%}")
        k4.metric("Mean risk probability", f"{avg_prob:.1%}")
    st.markdown("### How the system works")
    steps = st.columns(4)
    for col, number, title, desc in zip(steps, ["01", "02", "03", "04"], ["Collect", "Prepare", "Predict", "Support"], ["Weekly attendance and learning activity", "Validate columns and scale features", "Estimate a risk probability", "Review students and plan support"]):
        with col:
            st.markdown(f"**{number} · {title}**")
            st.write(desc)
    if isinstance(df, pd.DataFrame):
        st.markdown("### Current dataset preview")
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)
        st.caption(f"Showing 8 of {len(df)} records. Use Predict & Review or Student Insights for more analysis.")
    else:
        st.info("Load the synthetic demo data or upload a CSV to begin.")
    st.markdown("### Research note")
    st.write("This is a research prototype. Validate performance on genuine, institution-approved student data before drawing conclusions or deploying it for real interventions.")

with tabs[1]:
    st.subheader("Predict risk and review the student list")
    if not isinstance(df, pd.DataFrame):
        st.info("Load demonstration data or upload a CSV above to continue.")
    else:
        st.caption(f"Dataset: {len(df)} records · {len(df.columns)} columns")
        left, right = st.columns([1, 1])
        with left:
            if st.button("🧮 Run LSTM predictions", type="primary", use_container_width=True):
                if not (MODEL_PATH.exists() and SCALER_PATH.exists()):
                    st.error("No trained model found. Open Train & Evaluate and train a model first.")
                else:
                    try:
                        model = tf.keras.models.load_model(MODEL_PATH)
                        scaler = joblib.load(SCALER_PATH)
                        X, _, _, _, _ = prepare_sequences(df, require_target=False)
                        X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
                        probs = model.predict(X_scaled, verbose=0).ravel()
                        out = df.copy()
                        out["At_Risk_Probability"] = np.round(probs, 4)
                        out["Predicted_Category"] = np.where(probs >= threshold, "At-Risk", "Safe")
                        out["Suggested_Action"] = np.where(probs >= threshold, "Faculty review and supportive outreach", "Continue routine monitoring")
                        st.session_state["prediction_output"] = out
                        st.success("Predictions completed. Review the probabilities with appropriate context.")
                    except Exception as e:
                        st.error(f"Prediction failed. Check the CSV columns. Details: {e}")
        with right:
            if st.session_state.get("prediction_output") is not None:
                st.download_button("⬇️ Download prediction report", st.session_state["prediction_output"].to_csv(index=False).encode("utf-8"), file_name="student_risk_predictions.csv", mime="text/csv", use_container_width=True)
                try:
                    pdf_bytes = build_pdf_report(st.session_state["prediction_output"], threshold)
                    st.download_button("📄 Download summary report (PDF)", pdf_bytes, file_name="student_risk_summary_report.pdf", mime="application/pdf", use_container_width=True)
                except Exception as e:
                    st.warning(f"PDF report unavailable: {e}. The CSV report is still available.")
            else:
                st.button("Download report (run prediction first)", disabled=True, use_container_width=True)
        out = st.session_state.get("prediction_output")
        if isinstance(out, pd.DataFrame):
            total = len(out)
            risky = int((out["Predicted_Category"] == "At-Risk").sum())
            a, b, c, d = st.columns(4)
            a.metric("Students analysed", f"{total:,}")
            b.metric("Flagged At-Risk", f"{risky:,}")
            c.metric("Flagged share", f"{(risky / total * 100):.1f}%" if total else "0%")
            d.metric("Threshold", f"{threshold:.0%}")
            f1, f2 = st.columns([1, 2])
            with f1:
                counts = out["Predicted_Category"].value_counts().reindex(["At-Risk", "Safe"], fill_value=0)
                fig, ax = plt.subplots(figsize=(4, 3))
                counts.plot(kind="bar", ax=ax)
                ax.set_title("Predicted categories"); ax.set_xlabel(""); ax.set_ylabel("Students"); ax.tick_params(axis="x", rotation=0)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with f2:
                st.markdown("**Search, filter and review records**")
                id_candidates = [c for c in ["Student_ID", "student_id", "ID", "id"] if c in out.columns]
                search_text = st.text_input("Search student ID or text", key="student_search")
                category_filter = st.multiselect("Show categories", ["At-Risk", "Safe"], default=["At-Risk", "Safe"])
                min_prob = st.slider("Minimum probability shown", 0.0, 1.0, 0.0, 0.05, key="min_prob_filter")
                filtered = out[out["Predicted_Category"].isin(category_filter) & (out["At_Risk_Probability"] >= min_prob)].copy()
                if search_text.strip():
                    searchable = id_candidates[0] if id_candidates else out.columns[0]
                    filtered = filtered[filtered[searchable].astype(str).str.contains(search_text.strip(), case=False, na=False)]
                filtered = filtered.sort_values("At_Risk_Probability", ascending=False)
                st.dataframe(filtered, use_container_width=True, hide_index=True)
                st.caption(f"Showing {len(filtered)} of {len(out)} records. A flagged result is a prompt for supportive review, not a diagnosis or final decision.")

with tabs[2]:
    st.subheader("Student engagement insights")
    if not isinstance(df, pd.DataFrame):
        st.info("Load a dataset to explore student engagement patterns.")
    else:
        weekly_cols = [c for c in df.columns if str(c).lower().startswith("week")]
        if not weekly_cols:
            st.warning("No weekly feature columns were detected. Expected names such as week1_attendance_pct, week2_attendance_pct, etc.")
        else:
            st.markdown("**Dataset statistics**")
            numeric = df.select_dtypes(include=np.number)
            if not numeric.empty:
                st.dataframe(numeric.describe().T.round(2), use_container_width=True)
                left_chart, right_chart = st.columns(2)
                with left_chart:
                    st.markdown("**Feature distribution**")
                    numeric_feature = st.selectbox("Choose numeric feature", list(numeric.columns), key="distribution_feature")
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    ax.hist(pd.to_numeric(numeric[numeric_feature], errors="coerce").dropna(), bins=15, edgecolor="white")
                    ax.set_title(f"Distribution: {numeric_feature}"); ax.set_xlabel(numeric_feature); ax.set_ylabel("Record count")
                    st.pyplot(fig, use_container_width=True); plt.close(fig)
                with right_chart:
                    st.markdown("**Feature correlation**")
                    corr_cols = list(numeric.columns[:12])
                    corr = numeric[corr_cols].corr(numeric_only=True)
                    if not corr.empty:
                        fig, ax = plt.subplots(figsize=(6, 3.5))
                        image = ax.imshow(corr.fillna(0).values, aspect="auto", vmin=-1, vmax=1)
                        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
                        ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=7)
                        ax.set_title("Correlation matrix (up to 12 numeric columns)")
                        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
                        st.pyplot(fig, use_container_width=True); plt.close(fig)
            out_insights = st.session_state.get("prediction_output")
            if isinstance(out_insights, pd.DataFrame):
                st.markdown("**Risk distribution after prediction**")
                risk_counts = out_insights["Predicted_Category"].value_counts().reindex(["At-Risk", "Safe"], fill_value=0)
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.bar(risk_counts.index, risk_counts.values)
                ax.set_ylabel("Students"); ax.set_title("Predicted student categories")
                for i, value in enumerate(risk_counts.values): ax.text(i, value, str(value), ha="center", va="bottom")
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                st.download_button("⬇️ Download insights data (CSV)", out_insights.to_csv(index=False).encode("utf-8"), file_name="student_insights_and_predictions.csv", mime="text/csv")
            st.markdown("**Weekly feature trends**")
            features = sorted(set("_".join(str(c).split("_")[1:]) for c in weekly_cols if "_" in str(c)))
            chosen = st.selectbox("Choose a feature to inspect", features or weekly_cols)
            chosen_cols = [c for c in weekly_cols if str(c).endswith(chosen)] if features else weekly_cols
            if chosen_cols:
                def week_num(col):
                    import re
                    m = re.search(r"week(\d+)", str(col).lower())
                    return int(m.group(1)) if m else 999
                chosen_cols = sorted(chosen_cols, key=week_num)
                vals = [pd.to_numeric(df[c], errors="coerce").mean() for c in chosen_cols]
                labels = [f"Week {week_num(c)}" for c in chosen_cols]
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(labels, vals, marker="o")
                ax.set_title(f"Average {chosen.replace('_', ' ')} by week"); ax.set_ylabel("Mean value"); ax.grid(True, alpha=.25)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown("**Missing data check**")
            missing = df.isna().sum().rename("Missing values").to_frame()
            missing = missing[missing["Missing values"] > 0]
            if missing.empty:
                st.success("No missing values detected in the loaded dataset.")
            else:
                st.dataframe(missing, use_container_width=True)
            st.caption("Descriptive statistics show patterns in the uploaded data; they do not establish causation or prove model effectiveness.")

with tabs[3]:
    st.subheader("Train and evaluate the model")
    st.write("Train the compact LSTM and compare it with a Logistic Regression baseline. Training runs on your computer and may take several minutes.")
    st.warning("Use labelled data with an At_Risk column (1 = At-Risk, 0 = Safe). Synthetic demo metrics are only a software demonstration.")
    uploaded_train = st.file_uploader("Upload labelled training CSV (optional)", type=["csv"], key="train_csv")
    epochs = st.slider("Maximum training epochs", 3, 40, 15)
    if st.button("▶ Train lightweight LSTM", type="primary"):
        csv_path = ROOT / "data" / "uploaded_training_data.csv"
        if uploaded_train is not None:
            csv_path.parent.mkdir(exist_ok=True)
            csv_path.write_bytes(uploaded_train.getvalue())
        else:
            csv_path = ROOT / "data" / "synthetic_demo_data.csv"
            if not csv_path.exists():
                make_demo_data().to_csv(csv_path, index=False)
        with st.spinner("Training LSTM and Logistic Regression baseline..."):
            try:
                from train_model import train
                result = train(str(csv_path), epochs=epochs)
                st.success("Training completed. The values below were measured in this run.")
                st.dataframe(result, use_container_width=True, hide_index=True)
                st.download_button("Download model comparison metrics", result.to_csv(index=False).encode("utf-8"), "model_comparison.csv", "text/csv")
                st.caption("For a paper, report the dataset source, split strategy, hardware, and whether features or labels are simulated.")
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.code("python -m pip install -r requirements.txt\npython -m streamlit run app.py", language="bash")
    if RESULTS_PATH.exists():
        st.markdown("#### Latest saved comparison")
        try:
            saved = pd.read_csv(RESULTS_PATH)
            st.dataframe(saved, use_container_width=True, hide_index=True)
        except Exception as e:
            st.caption(f"Saved results could not be displayed: {e}")

with tabs[4]:
    st.subheader("Model evaluation and explainability")
    pred_path = ROOT / "results" / "test_predictions.csv"
    importance_path = ROOT / "results" / "feature_importance.csv"
    comparison_path = ROOT / "results" / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.markdown("### LSTM versus Logistic Regression")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.download_button("Download comparison metrics CSV", comparison.to_csv(index=False).encode("utf-8"),
                           file_name="model_comparison.csv", mime="text/csv")
    if pred_path.exists():
        test = pd.read_csv(pred_path)
        if {"actual", "lstm_probability", "lstm_prediction"}.issubset(test.columns) and len(test):
            y_true = test["actual"].astype(int).to_numpy()
            y_prob = test["lstm_probability"].astype(float).to_numpy()
            y_pred = test["lstm_prediction"].astype(int).to_numpy()
            left_eval, right_eval = st.columns(2)
            with left_eval:
                st.markdown("### Confusion matrix")
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xticks([0, 1], labels=["Safe", "At-Risk"])
                ax.set_yticks([0, 1], labels=["Safe", "At-Risk"])
                ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label")
                for i in range(2):
                    for j in range(2): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
                fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
            with right_eval:
                st.markdown("### ROC curve")
                if len(np.unique(y_true)) == 2:
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots(figsize=(4.5, 3.5))
                    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
                    ax.plot([0, 1], [0, 1], linestyle="--", label="No-skill reference")
                    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
                    ax.legend(loc="lower right"); ax.grid(alpha=.2)
                    st.pyplot(fig, use_container_width=True); plt.close(fig)
                else: st.info("ROC curve needs both classes in the test split.")
            st.download_button("Download test predictions CSV", test.to_csv(index=False).encode("utf-8"),
                               file_name="test_predictions.csv", mime="text/csv")
        else:
            st.info("Saved test predictions do not contain the expected evaluation columns yet. Train the model again.")
    else:
        st.info("Train the model first to generate a confusion matrix and ROC curve.")
    st.markdown("### Explainable AI: permutation feature importance")
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        if not importance.empty and "importance_auc_drop" in importance:
            importance = importance.sort_values("importance_auc_drop", ascending=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(importance["feature"], importance["importance_auc_drop"])
            ax.set_xlabel("ROC-AUC drop after permutation"); ax.set_title("Global feature importance")
            ax.grid(axis="x", alpha=.2)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.dataframe(importance, use_container_width=True, hide_index=True)
            st.download_button("Download feature importance CSV", importance.to_csv(index=False).encode("utf-8"),
                               file_name="feature_importance.csv", mime="text/csv")
            st.caption("Permutation importance is a global model-behaviour measure, not a causal explanation or a guaranteed explanation for an individual student.")
        else:
            st.info("No feature importance results are available. Retrain the model to calculate them.")
    else:
        st.info("Train the model to calculate test-set permutation feature importance.")
    st.markdown("### Resource efficiency")
    if comparison_path.exists():
        comp = pd.read_csv(comparison_path)
        lstm_row = comp[comp["model"] == "Lightweight LSTM"]
        if not lstm_row.empty:
            row = lstm_row.iloc[0]
            a, b, c = st.columns(3)
            a.metric("Training time", f"{row.get('training_time_seconds', float('nan')):.2f} s" if pd.notna(row.get('training_time_seconds')) else "—")
            b.metric("Model size", f"{row.get('model_size_mb', float('nan')):.2f} MB" if pd.notna(row.get('model_size_mb')) else "—")
            c.metric("CPU inference / sample", f"{row.get('cpu_inference_ms_per_sample', float('nan')):.3f} ms" if pd.notna(row.get('cpu_inference_ms_per_sample')) else "—")
            st.caption("Efficiency figures are measurements from the latest local run and may vary by hardware.")

with tabs[5]:
    st.subheader("CSV format and instructions")
    st.markdown("Training data must contain a binary **At_Risk** target and the same feature names for each week. Prediction data do not need the target column.")
    st.code("Student_ID,week1_attendance_pct,week1_lms_logins,week1_resource_accesses,week1_submission_delay_days,week1_assessment_score,...,At_Risk\nS001,82,5,12,1,68,...,0\nS002,45,1,3,6,32,...,1", language="csv")
    st.markdown("**Expected weekly features**")
    st.table(pd.DataFrame({"Feature": ["attendance_pct", "lms_logins", "resource_accesses", "submission_delay_days", "assessment_score"], "Meaning": ["Attendance percentage", "Learning platform logins", "Learning resource access count", "Days late for submission", "Assessment score"]}))
    st.markdown("**Run locally on Windows**")
    st.code("python -m venv venv\nvenv\\Scripts\\activate\npython -m pip install -r requirements.txt\npython data_simulator.py\npython -m streamlit run app.py", language="bash")
    st.caption("Keep the terminal open while using the app. Open http://localhost:8501 in your browser.")
