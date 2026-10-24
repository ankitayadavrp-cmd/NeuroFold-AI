import streamlit as st
import pandas as pd

st.set_page_config(page_title="EnergyBridge AI", page_icon="♻️", layout="wide")

# EnergyBridge AI is an early decision-support prototype. Estimates are illustrative,
# based on transparent assumptions, and must not be presented as engineering guarantees.

st.markdown("""
<style>
.main {background-color: #f7faf7;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#12372a,#28745b); color: white;}
.hero h1 {color:white; margin-bottom:0.2rem;}
.hero p {color:#e7f4ed; font-size:1.05rem;}
.small-note {color:#5b665f; font-size:0.88rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>♻️ EnergyBridge AI</h1>
<p>Turning local waste challenges into practical clean-energy opportunities.</p>
</div>
""", unsafe_allow_html=True)

st.caption("A prototype for community-level waste assessment and renewable-energy decision support.")

with st.sidebar:
    st.header("Assess a waste stream")
    location = st.text_input("Community / location", "Example: Mohali, Punjab")
    waste_type = st.selectbox(
        "Main waste type",
        ["Agricultural residues", "Animal manure", "Food / market waste",
         "Municipal mixed waste", "Plastic waste", "Wastewater / sewage sludge"]
    )
    quantity = st.number_input("Available waste (tonnes per day)", min_value=0.0, max_value=10000.0, value=2.0, step=0.5)
    moisture = st.selectbox("Condition (if known)", ["Not sure", "Mostly wet", "Mostly dry", "Mixed"])
    assess = st.button("Assess opportunity", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("**Prototype note**")
    st.caption("Recommendations are screening-level guidance, not a feasibility study. Local composition, contamination, logistics and regulations must be checked.")

if "assessment" not in st.session_state:
    st.session_state.assessment = None
if assess:
    st.session_state.assessment = {
        "location": location.strip() or "Location not specified",
        "waste_type": waste_type,
        "quantity": quantity,
        "moisture": moisture
    }

st.subheader("How the platform works")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 1 · Assess")
    st.write("Describe the waste stream and its approximate daily quantity.")
with c2:
    st.markdown("### 2 · Match")
    st.write("Screen potential pathways based on waste type.")
with c3:
    st.markdown("### 3 · Connect")
    st.write("Identify the types of local expertise needed to evaluate implementation.")

if st.session_state.assessment:
    a = st.session_state.assessment
    st.markdown("---")
    st.header("Opportunity assessment")
    st.write(f"**Location:** {a['location']}  |  **Waste stream:** {a['waste_type']}  |  **Quantity:** {a['quantity']:,.1f} tonnes/day")

    pathways = {
        "Agricultural residues": {
            "best": "Feedstock assessment for anaerobic digestion or other residue valorization",
            "why": "Some residues can be used in bioenergy systems, but suitability depends on crop type, lignin content, moisture, collection logistics and competing uses.",
            "partners": ["Agricultural extension service", "Biogas / biomass technology provider", "Farmer producer organization", "Local energy or waste-management department"],
            "caution": "Do not assume all crop residues are suitable for digestion; some are better suited to other processing routes."
        },
        "Animal manure": {
            "best": "Anaerobic digestion screening for biogas and digestate",
            "why": "Manure can be a candidate feedstock for anaerobic digestion when collection, dilution, system design and digestate management are appropriate.",
            "partners": ["Biogas plant designer", "Veterinary / livestock extension service", "Local panchayat or cooperative", "Digestate / nutrient-management specialist"],
            "caution": "Actual gas yield depends on manure characteristics, system temperature, retention time and operating practice."
        },
        "Food / market waste": {
            "best": "Source-separated organic waste assessment for anaerobic digestion or composting",
            "why": "Clean, separated food waste may be suitable for biogas production; composting can be an alternative depending on scale and local demand.",
            "partners": ["Market committee / municipal body", "Organic-waste collector", "Biogas or composting operator", "Environmental engineer"],
            "caution": "Contamination with plastic, chemicals or other non-organic material can reduce suitability."
        },
        "Municipal mixed waste": {
            "best": "Waste characterization and segregation study before selecting a pathway",
            "why": "Mixed waste should not be assigned a single conversion technology without composition testing and segregation. Organic fractions and recyclable fractions need separate handling.",
            "partners": ["Municipal waste department", "Waste characterization laboratory", "Material recovery facility", "Environmental engineer"],
            "caution": "This prototype does not recommend direct processing of mixed waste without a professional assessment."
        },
        "Plastic waste": {
            "best": "Segregation, reuse and authorized recycling pathway assessment",
            "why": "Plastic waste generally needs polymer identification, clean segregation and a suitable authorized recycling route; it is not an appropriate feedstock for a basic biogas estimate.",
            "partners": ["Authorized plastic recycler", "Material recovery facility", "Producer-responsibility organization", "Municipal waste department"],
            "caution": "No biogas or energy-yield estimate is generated for plastic waste."
        },
        "Wastewater / sewage sludge": {
            "best": "Professional wastewater and sludge characterization for treatment and biogas recovery potential",
            "why": "Some sludge streams can support anaerobic digestion, but pathogen control, contaminants, process design and regulatory requirements are critical.",
            "partners": ["Wastewater utility", "Environmental engineering consultant", "Accredited testing laboratory", "Authorized treatment-plant operator"],
            "caution": "Requires specialist sampling, treatment design and compliance review."
        }
    }

    p = pathways[a["waste_type"]]
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("### Recommended next step")
        st.success(p["best"])
        st.write(p["why"])
        st.markdown("### Potential implementation partners")
        for partner in p["partners"]:
            st.write(f"• {partner}")
    with right:
        st.markdown("### Screening summary")
        st.metric("Reported waste quantity", f"{a['quantity']:,.1f} t/day")
        if a["waste_type"] in ["Animal manure", "Food / market waste"]:
            st.metric("Energy estimate", "Not calculated")
            st.caption("A reliable estimate requires feedstock-specific yield data and process assumptions. The prototype deliberately avoids inventing a gas-yield number.")
        elif a["waste_type"] == "Agricultural residues":
            st.metric("Energy estimate", "Needs feedstock data")
            st.caption("Residue type and conversion pathway must be specified before calculating energy potential.")
        else:
            st.metric("Energy estimate", "Not applicable yet")
            st.caption("The first step is characterization, segregation or specialist review.")
        st.info("**Key uncertainty:** " + p["caution"])

    st.markdown("### Suggested action plan")
    plan = pd.DataFrame([
        {"Step": "1", "Action": "Confirm quantity and seasonal availability", "Responsible party": "Waste generator / community"},
        {"Step": "2", "Action": "Check composition and contamination", "Responsible party": "Qualified testing or technical partner"},
        {"Step": "3", "Action": "Compare technical and financial feasibility", "Responsible party": "Technology provider / engineer"},
        {"Step": "4", "Action": "Review permits, safety and end-use needs", "Responsible party": "Local authorities and project partners"},
    ])
    st.dataframe(plan, hide_index=True, use_container_width=True)

    st.download_button(
        "Download assessment summary (CSV)",
        data=pd.DataFrame([{
            "location": a["location"],
            "waste_type": a["waste_type"],
            "quantity_tonnes_per_day": a["quantity"],
            "condition": a["moisture"],
            "screening_recommendation": p["best"],
            "next_step": "Characterize feedstock and conduct a professional feasibility assessment"
        }]).to_csv(index=False),
        file_name="energybridge_assessment.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.info("Choose a waste stream in the sidebar and click **Assess opportunity** to generate a sample assessment.")
    st.markdown("### Try a demonstration scenario")
    st.write("For your video, try **Animal manure** or **Food / market waste** with a quantity of **2 tonnes/day**. Explain that the prototype screens options and identifies next steps; it does not yet verify a project's energy output or actual partner availability.")

st.markdown("---")
st.markdown("**EnergyBridge AI · Early-stage prototype**")
st.markdown('<p class="small-note">This tool provides preliminary educational decision support only. It does not replace laboratory characterization, engineering design, financial analysis, environmental assessment or regulatory advice. Partner categories are suggestions, not confirmed providers. No AI/ML model is currently claimed in this version.</p>', unsafe_allow_html=True)
