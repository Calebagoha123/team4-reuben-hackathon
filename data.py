"""Mock patient record for the demo EHR.

The chart starts EMPTY on purpose — the whole point of the tool is to populate
it from a photographed handwritten note. So the clinical lists are blank and the
demographics are unset; the templates render em-dash / empty-state placeholders.
The field *structure* below is what the UI lays out (and, next step, what the
model fills in).
"""

PATIENT = {
    "name": "",
    "mrn": "",
    "dob": "",
    "age": "",
    "sex": "",
    "location": "",
    "provider": "",
    # Facesheet — all empty until data is entered / scanned in
    "problems": [],       # {name, onset, status}
    "medications": [],    # {name, sig, prescriber}
    "allergies": [],      # {allergen, reaction}
    "orders": [],         # {date, name, status}
    "vitals": [],         # {date, bp, hr, temp, rr, spo2, wt}
    "family_hx": "",
    "social_hx": "",
    "surgical_hx": "",
    "tasks": [],          # {task, due, status}
}

# The blank medical-note schema. Order matters — this is the on-screen form,
# and (next step) the set of fields the model will populate from a photo.
NOTE_FIELDS = [
    ("note_type", "Note Type", "input"),
    ("chief_complaint", "Chief Complaint", "input"),
    ("hpi", "History of Present Illness (HPI)", "textarea"),
    ("pmhx", "Past Medical History (PMHx)", "textarea"),
    ("fmhx", "Family History (FMHx)", "textarea"),
    ("shx", "Social History (SHx)", "textarea"),
    ("ros", "Review of Systems (ROS)", "textarea"),
    ("pe", "Physical Exam (PE)", "textarea"),
    ("assessment", "Assessment", "textarea"),
    ("plan", "Plan", "textarea"),
]
