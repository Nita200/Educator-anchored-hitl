from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="Nita200/educator-anchored-hitl-pubmedbert",
)

result = clf(
    "Patient presents with sudden onset crushing chest pain radiating to the left arm," \
    " diaphoresis, and shortness of breath. Student judgment: This is likely just anxiety"
    "and the patient should be reassured and sent home. [SEP] Rationale: The patient appears " \
    "stressed and chest tightness is common with anxiety, so no further cardiac workup is needed."
)
print(result)