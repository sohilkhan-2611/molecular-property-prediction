"""Phase 15: a small web page around predict.py.

Run:  streamlit run app.py          (opens http://localhost:8501 in your browser)

Needs   artifacts/final_model.pt    create it once with:  python src/train_final.py

The page and the command line share the same functions in predict.py, so they cannot disagree.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from predict import CHECKPOINT, load_checkpoint, predict_smiles, score_many, typical_error_line  # noqa: E402

BENCH_CSV = ROOT / "results" / "phase13_benchmark.csv"
MAX_ROWS = 2000
EXAMPLES = {
    "Ethanol": "CCO",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
}

st.set_page_config(page_title="XLogP3 predictor", page_icon="🧪", layout="centered")


@st.cache_resource(show_spinner="Loading the model...")
def get_model():
    return load_checkpoint()


def set_example(smiles):
    st.session_state["smiles_input"] = smiles       # runs before the text box is drawn again


def sentence(text):
    """Capitalise the first letter only (str.capitalize would lowercase MAE)."""
    return text[:1].upper() + text[1:]


def show_molecule(canonical):
    mol = Chem.MolFromSmiles(canonical)
    st.image(Draw.MolToImage(mol, size=(380, 280)), caption=canonical)


# ----------------------------------------------------------------------
# Header and model loading
# ----------------------------------------------------------------------
st.title("Molecule XLogP3 predictor")
st.caption("Type a SMILES string and get PubChem's computed XLogP3, as imitated by a graph neural network. "
           "This is a model of a computed number, not a lab measurement.")

if not CHECKPOINT.exists():
    st.error("No trained model found. Create it once in your terminal with:  python src/train_final.py")
    st.stop()
ck, nets = get_model()

tab_one, tab_many, tab_about = st.tabs(["One molecule", "Many molecules", "About this model"])

# ----------------------------------------------------------------------
# Tab 1: one molecule
# ----------------------------------------------------------------------
with tab_one:
    st.session_state.setdefault("smiles_input", "CCO")
    st.text_input("SMILES", key="smiles_input", help="For example CCO is ethanol, c1ccccc1 is benzene.")
    cols = st.columns(len(EXAMPLES))
    for col, (name, smi) in zip(cols, EXAMPLES.items()):
        col.button(name, on_click=set_example, args=(smi,))

    query = st.session_state["smiles_input"].strip()
    if query:
        r = predict_smiles(query, nets, ck)
        if not r["ok"]:
            st.error(f"Could not predict: {r['message']}.")
        else:
            left, right = st.columns([1, 1])
            with left:
                st.metric("Predicted XLogP3", f"{r['prediction']:.2f}")
                if len(nets) > 1:
                    st.caption(f"The {len(nets)} networks differ by about {r['spread']:.2f}.")
                st.caption("Higher means more fat loving, lower means more water loving. Ethanol sits near 0, "
                           "benzene above 2.")
                for note in r["notes"]:
                    st.warning(sentence(note) + ".")
            with right:
                show_molecule(r["canonical"])
            if r["canonical"] != query:
                st.caption(f"Standardised from your input to {r['canonical']} (stereochemistry and labels removed).")
        line = typical_error_line(ck)
        if line:
            st.caption(sentence(line) + ".")

# ----------------------------------------------------------------------
# Tab 2: many molecules
# ----------------------------------------------------------------------
with tab_many:
    st.write("Paste one SMILES per line, or upload a CSV that has a SMILES column.")
    pasted = st.text_area("Paste SMILES", height=120, placeholder="CCO\nc1ccccc1\nCC(=O)Oc1ccccc1C(=O)O")
    upload = st.file_uploader("or upload a CSV", type="csv")

    smiles_list = [ln.strip() for ln in pasted.splitlines() if ln.strip()]
    source = "pasted list"
    if upload is not None:
        try:
            df = pd.read_csv(upload)
        except Exception as exc:                                  # an unreadable file should not crash the page
            st.error(f"Could not read that CSV: {exc}")
            df = None
        if df is not None and len(df.columns):
            guess = next((i for i, c in enumerate(df.columns) if str(c).strip().lower() in ("smiles", "smile")), 0)
            column = st.selectbox("Which column holds the SMILES?", list(df.columns), index=guess)
            smiles_list = df[column].tolist()
            source = f"column '{column}' of {upload.name}"

    if smiles_list and st.button("Predict all", type="primary"):
        if len(smiles_list) > MAX_ROWS:
            st.warning(f"Only the first {MAX_ROWS} of {len(smiles_list)} rows are scored.")
            smiles_list = smiles_list[:MAX_ROWS]
        bar = st.progress(0.0, text=f"Scoring {len(smiles_list)} molecules from the {source}")
        table = score_many(smiles_list, nets, ck, on_progress=lambda f: bar.progress(f))
        bar.empty()
        st.session_state["batch"] = table              # kept, so the download button does not erase it

    if "batch" in st.session_state:
        table = st.session_state["batch"]
        bad = int((table["status"] != "ok").sum())
        st.write(f"{len(table) - bad} predicted, {bad} rejected.")
        st.dataframe(table, hide_index=True)
        st.download_button("Download results as CSV", table.to_csv(index=False).encode("utf-8"),
                           file_name="xlogp3_predictions.csv", mime="text/csv")

# ----------------------------------------------------------------------
# Tab 3: about
# ----------------------------------------------------------------------
with tab_about:
    names = {"gcn": "graph convolutional network (GCN)", "mpnn": "message passing network with bond features (MPNN)",
             "mpnn_nobond": "message passing network without bond features"}
    st.markdown(
        f"**Model:** {names.get(ck['model'], ck['model'])}, an average of {len(nets)} networks.  \n"
        f"**Trained on:** {ck['n_train']} molecules from PubChem, target = PubChem's computed {ck['target'].split()[-1]}.  \n"
        "**Pipeline:** SMILES, cleaned with the same rules as the training data, turned into a graph of atoms and "
        "bonds, then predicted by the networks.")
    line = typical_error_line(ck)
    if line:
        st.info(sentence(line) + ". Measured on held-out molecules with the same kind of network. This final model "
                "trained on all molecules, so it has no test set of its own.")
    st.markdown(
        "**Know the limits**\n"
        "- The target is a computed number from PubChem, not an experimental logP.\n"
        "- Very small molecules are rare in the training data, so errors of 0.5 or more are possible there.\n"
        "- Only the elements C, H, N, O, F, P, S, Cl, Br, I and single molecules up to 1000 g/mol are accepted.\n"
        "- The spread between networks is a warning light, not a confidence interval.")
    if BENCH_CSV.exists():
        b = pd.read_csv(BENCH_CSV)
        b = b[~b["kind"].isin(["ensemble"]) & ~b["kind"].str.startswith("blend")]
        wide = b.pivot_table(index="model", columns="split", values="MAE").sort_values("scaffold")
        st.write("Test error (MAE, lower is better) of every model in this project:")
        st.dataframe(wide.round(3).rename(columns={"random": "random split", "scaffold": "scaffold split"}))
