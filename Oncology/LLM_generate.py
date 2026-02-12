from multiprocessing import parent_process
import transformers
import torch
import argparse
import random
import json
import gc
import os
import pickle
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpt5_thinking_model import GPT5Model, GPT5Config

TASK_TAG = "directedTBD" # directedTBD or fullAPTBD/fullAP_TBD or CoT

import re

torch.set_float32_matmul_precision('high')
parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model', help='model type (openbiollm, medgemma, gpt-5)', type=str)
parser.add_argument('--prompt', dest='prompt', help='SOAP case', type=str)
parser.add_argument('--reasoning_effort', dest='reasoning_effort', 
                   help='Reasoning effort for GPT-5 thinking model (minimal/low/medium/high)', 
                   type=str, default='medium')
parser.add_argument('--verbosity', dest='verbosity', 
                   help='Verbosity level for GPT-5 thinking model (low/medium/high)', 
                   type=str, default='medium')
parser.add_argument('--openai_api_key', dest='openai_api_key', 
                   help='OpenAI API key for GPT-5 thinking model', 
                   type=str, default=None)
parser.add_argument('--batch_size', dest='batch_size', 
                   help='Batch size for HF models (openbiollm/medgemma). Ignored for gpt-5.', 
                   type=int, default=1)
parser.add_argument('--max_output_tokens', dest='max_output_tokens', 
                   help='Maximum output tokens for GPT-5 model', 
                   type=int, default=16384)

args = parser.parse_args()


MODEL = args.model
PROMPT = args.prompt
REASONING_EFFORT = args.reasoning_effort
VERBOSITY = args.verbosity
OPENAI_API_KEY = args.openai_api_key
BATCH_SIZE = args.batch_size
MAX_OUTPUT_TOKENS = args.max_output_tokens

S_O_Squamous_Lung_routineFollowup = \
    """
    \"Subjective:
    This a 58 y.o. male with discoid lupus, extensive smoking history (40 pack-years, quit 2019), and newly diagnosed squamous cell carcinoma of the lung is here for further management. He was initially hospitalized for hemoptysis on 12/21/23 locally. CT on 12/23/23 showing LLL completely atelectatic, small left pleural effusion, mildly enlarged AP node, small bilateral adrenal lesions (1.5 cm on the right with hounsfield of 7, and 1.3 cm on the left with Hounsfield 20) concern for adenoma.  12/29/23: Bronch, distal left main bronchus, just prior to left upper lobe, a fungating bleeding mass was visualized, with 60-70% obstruction to the airway. This lesion was biopsied. Per report, no abnormal LN station were seen. Biopsy positive for squamous cell carcinoma. He also had an arterial embolization done on 12/26/23.

    Interval History
    Patient is here for his initial follow at OSU.
    Overall he is feeling relatively well. Since the initial hospitalization at the end of December, he has not had further hemoptysis. Denies any significant chest pain. Has some intermittent mild cough. Overall, he is feeling stable.  
    He has a history of discoid lupus involving the face and chest skin, on plaquenil for 6 months back in 1980s. 
    He denies any headache, vision changes, hearing changes, dysphagia, nausea, vomiting, abd pain, constipation, diarrhea, dysuria, focal weakness, leg swelling, peripheral neuropathy, fever, chill. 


    Objective:
    Vital Signs: BP 160/73 (BP Location: Right arm, BP Position: Sitting)  | Pulse 78  | Temp 97.7 °F (36.5 °C) (Oral)  | Resp 14  | Ht 1.829 m (6')  | Wt 127 kg (280 lb) Comment: with shoes and jacket on | SpO2 95%  | BMI 37.97 kg/m²  | Smoking Status Former 
    ECOG performance status: 0
    Constitutional:  male alert and oriented in no acute distress.  
    Eyes: No scleral icterus. EOMI.
    ENMT: No oropharyngeal lesions or thrush. 
    Neck: no masses, symmetrical. 
    Respiratory: Normal effort. Clear to auscultation: A&P bilaterally, no crackles/rhonchi/wheezes
    Cardiovascular: Normal S1/S2, regular rate and rhythm, no murmurs, gallops, rubs. No peripheral edema bilaterally
    Abdomen: soft, non-tender. Bowel sounds normal. 
    Musculoskeletal: no digital cyanosis or clubbing.  
    Skin: No rashes or lesions. 
    Lymph nodes:  Cervical and supraclavicular nodes normal.
    Neurologic: Cranial nerves grossly intact. No focal deficits. 
    Psychosocial: Affect appropriate for situation. Alert and oriented to person, place and time. 

    12/29/23: Bronch with left main bronchus mass biopsy positive for squamous cell carcinoma. 

    CT Chest:
    12/23/23:
    1. The left lower lobe is completely atelectatic and there is abrupt 
    termination of the proximal left lower lobe bronchus concern for obstructive from a mass. 
    2. mildly enlarged mediastinal lymph node.
    Small bilateral adrenal lesion.
    
    CT Abdomen:
    None
    
    MRI Brain: 
    None
    
    PET/CT:
    None\"
    """

S_O_Squamous_Lung_initiateTreatment =\
    """
    Subjective:
    This is a 60 y.o. male, former heavy smoker, with newly diagnosed metastatic carcinoma, favoring squamous cell carcinoma from pulmonary origin is here for further management. He initially presented in the ED on 4/24/25 for severe left flank pain and was found to have spontaneous subcapsular renal hematoma. CT on 4/24/25 significant for moderate sized left subcapsular hematoma measuring 2.5 cm, and right renal cortical cysts measuring 2.5 cm; additionally, there is a 1.7 cm probable liver cyst, and a right 1.6 cm adrenal nodule. Follow up MRI on 4/30/25 showing exophytic simple cyst in the right kidney, and heterogenous enhancing lesion in the left upper pole of left kidney with hematoma; enhancing 2.3 cm lesion in the liver. CT chest on 5/5/25 showing right upper lobe mass measuring 4.7 cm, mediastinal and hilar lymphadenopathy with scattered pulmonary nodules. A liver biopsy was done on 5/7/25, showing poorly differentiated carcinoma (PD-L1 99% in house, and 60% from Tempus; CPS 60 from tempus, +TP53, NOTCH2, and NF2). He presented in the ED again for severe right flank pain, and was found to have bilateral subcapsular hematoma in both kidney, with AKI that required brief dialysis. 
    
    PET scan as part of workup for carcinoma of unknown primary on 5/22/25 showing significantly increased uptake in the left tonsils, left lung mass and thoracic lymph nodes, liver. EGD and colonoscopy was done, few polyps (tubular adenomas and hyperplastic polyps) were seen and negative for malignancy. 
    
    A CT guided lung biopsy was done on 5/25/25 for the right upper lobe lung mass. Positive for non-small cell carcinoma, favoring squamous cell carcinoma (TPS 60%, tumor morphology is similar to that liver biopsy). Tempus tissue origin was run from the liver sample, 46% probability being pancreatic cancer and 11% pulmonary squamous cell carcinoma, and 11% probability from head and neck. 
    
    Interval History
    Patient is here with his wife today for follow up.
    
    Overall, he is feeling better since discharged from the hospital. He is able to ambulate without difficulty. Pulmonary symptoms is stable, without significant worsening of chest pain, dyspnea, hemoptysis. 
    
    He still has difficulty with urinary and still has indwelling foley catheter. 
    
    He denies any headache, vision changes, hearing changes, dysphagia, nausea, vomiting, abd pain, constipation, diarrhea, dysuria, focal weakness, leg swelling, peripheral neuropathy, fever, chill. 
    
    Objective:
    Vital Signs: BP 130/78 (BP Location: Left arm, BP Position: Sitting)  | Pulse 70  | Temp 98.2 °F (36.8 °C) (Oral)  | Resp 18  | Ht 1.803 m (5' 11") Comment: reported by pt | Wt 100.2 kg (221 lb) Comment: with shoes on | SpO2 98%  | BMI 30.82 kg/m²  | Smoking Status Heavy Smoker 
    ECOG performance status: 1
    Constitutional:  male alert and oriented in no acute distress.  
    Eyes: No scleral icterus. EOMI.
    ENMT: No oropharyngeal lesions or thrush. 
    Neck: no masses, symmetrical. 
    Respiratory: Normal effort. Clear to auscultation: A&P bilaterally, no crackles/rhonchi/wheezes
    Cardiovascular: Normal S1/S2, regular rate and rhythm, no murmurs, gallops, rubs. No peripheral edema bilaterally
    Abdomen: soft, non-tender. Bowel sounds normal. 
    Musculoskeletal: no digital cyanosis or clubbing.  
    Skin: No rashes or lesions. 
    Lymph nodes:  Cervical and supraclavicular nodes normal.
    Neurologic: Cranial nerves grossly intact. No focal deficits. 
    Psychosocial: Affect appropriate for situation. Alert and oriented to person, place and time.

    Pathology:
    5/25/25: CT guided biopsy of the lung mass
    Non-small cell carcinoma, favor squamous cell carcinoma

    5/7/25:
    Pathologic Diagnosis A. Liver, lesion, biopsy: · Poorly differentiated carcinoma


    CT chest
    5/5/25
    4.3 CM right upper lobe mass. Scattered pulmonary nodules. Enlarged mediastinal lymph nodes. 

    CT abd pelvis
    5/23/25:
    Known progressing pulmonary metastases. Stable subcapsular renal hematoma bilaterally. 

    PET/CT
    5/22/25
    Significantly increased uptake in the left tonsils, left lung mass and thoracic lymph nodes, liver.

    Brain MRI
    5/5/25
    No metastatic disease
    """

sys_prompt_Squamous_Lung_init = \
    """
    You are a leading expert physician in Medical Oncology. Your goal is to formulate the exact clinical stage diagnosis if possible and the best definitive plan for your patient's visit, using the provided Subjective and Objective note from your patient encounter. 
    Please keep in mind the patient encounter note may be missing pieces of information. If the patient encounter note contains [MISSING INFO], then you can assume that information is missing. For phrases where that is not present, missing information may or may not be present. Regardless, you are still responsible for fully understanding the note. Your patient encounter note (Subjective and Objective) is summarized below:
    """
sys_prompt_Squamous_Lung_end = \
    """
    You need to be very specific in your response, providing your rationale and using medical terminology where appropriate. 
    """


if TASK_TAG == "fullAP_TBD" or TASK_TAG == "fullAPTBD":
    user_prompt_Squamous_Lung_routineFollowup = \
        """
        Based on your goal and this note, 
        1.	Provide a detailed clinical assessment
        2.	Provide a detailed clinical plan based on your assessment
        3.	Identify the best option for definite management if applicable.
        4.	Would you like to perform any additional tests? Only suggest additional tests if you think it is strictly necessary in significantly informing your plan.
        Draft your response using the A&P style format note I have provided below. Strictly fill out and adhere to the instructions provided in the areas enclosed in “[]”.

        NOTE TO FILL OUT:
        \"\"\"
        A&P:
        [Patient summary of presentation]. 
        
        Abbreviated Plan Summary
        [Abbreviated Plan Summary, write "unknown" anywhere where the answer is not known.]
        •	Stage: [Stage, If this cannot be determined from the provided information, write "unknown". If you think that the staging is "unknown", write "TBD" if it is pending completion. Otherwise, strictly write the exact TNM staging.]
        •	Primary disease site: [Primary disease site]
        •	Histology: [Histology diagnosis]
        •	Significant Molecular Status: [Significant Molecular Status]
        •	PD-L1%: [PD-L1%]
        •	Prior cancer therapies: [Prior cancer therapies]
        •	Last staging (imaging): [Last staging (imaging) date and status]
        •	Current Therapy: [Current therapy]
        •	Anticipate next treatment options: [Anticipate next treatment options. If to be decided, write TBD.]
        
        1.	[Additional imaging only if strictly necessary]. 
        2.	[Additional testing only if strictly necessary]. 
        3.	[What you would like to discuss with the patient]. 
        4.	If no metastatic disease is identified, will [plan if no metastatic disease found]. 
        5.	[return to clinic timeframe in weeks]. 
        
        [The other disease of the patient]
        -[description of disease manifestation on patient]
        -[tried treatments by patient].
        -[referral plan]. 
        \"\"\"
        """
    
    user_prompt_Squamous_Lung_initiateTreatment = \
        """
        Based on your goal and this note, 
        1.	Provide a detailed clinical assessment
        2.	Provide a detailed clinical plan based on your assessment
        3.	Identify the best option for definite management if applicable.
        4.	Would you like to perform any additional tests? Only suggest additional tests if you think it is strictly necessary in significantly informing your plan.
        Draft your response using the A&P style format note I have provided below. Strictly fill out and adhere to the instructions provided in the areas enclosed in “[]”.

        NOTE TO FILL OUT:
        \"\"\"
        A&P:
        [Patient summary of presentation]. 

        Abbreviated Plan Summary
        [Abbreviated Plan Summary, write "unknown" anywhere where the answer is not known.]
        •	Stage: [Clinical Stage, If this cannot be determined from the provided information, write "unknown". If you think that the staging is "unknown", write "TBD" if it is pending completion. Otherwise, strictly write the exact TNM staging.]
        •	Primary disease site: [Primary disease site]
        •	Histology: [Histology diagnosis]
        •	Significant Molecular Status: [Significant Molecular Status]
        •	PD-L1%: [PD-L1%]
        •	Prior cancer therapies: [Prior cancer therapies]
        •	Last staging (imaging): [Last staging (imaging) date and status]
        •	Current Therapy: [Current therapy]
        •	Anticipate next treatment options: [Anticipate next treatment options. If to be decided, write TBD.]
        
        1.	[Identify likely origin of cancer and diagnosis. Use the imaging and lab results to reach your conclusion, describing how you used these findings to arrive at your answer.]
        2.	[Best definitive management therapy option. Use the necessary inmagineg and lab results to reach your conclusion, describing how you used these findings to arrive at your answer.]. 
        3.	[Identify the Keynote Study that supports your choice for definitive management option. Describe this study's key findings.] 
        4.	Due to concern for long commute, he is scheduled to see Dr K locally [return to clinic timeframe in weeks] for treatment.
        
        Spontaneous subcapsular hematoma of both kidneys:
        -[brief summary of resulting issue]
        -I have discussed the case with Dr G from GU oncology. Overall, [amount of concern for RCC, either low, medium, or high] concern for RCC currently. However, will benefit from [identify what might help evaluate concern] in the future if feasible
        -[follow up plan] 
        
        Urinary obstruction:
        -[follow up plan]
        -[what treatment the patient currently has for this] 
        \"\"\"
        """
elif TASK_TAG == "directedTBD":
    user_prompt_Squamous_Lung_routineFollowup = \
        """
        please only tell me:
        1. The best definitive management option for the patient if applicable
        
        Write "unknown" anywhere where the answer is not known.
        """
    user_prompt_Squamous_Lung_initiateTreatment = \
        """
        please only tell me:
        1. The best definitive management option for the patient if applicable
        
        Write "unknown" anywhere where the answer is not known.
        """
# The exact clinical staging for the patient. If this cannot be determined from the provided information, write "unknown". If you think that the staging is "unknown", write "TBD" if it is pending completion. Otherwise, strictly write the exact clinical staging.
# Would you like to perform any additional tests? Only suggest additional tests if you think it is strictly necessary in significantly informing your definitive management plan.

S_O_UC_var = \
    """
\"Subjective: Patricia Gateau is a 61-year-old female with PMHx significant for HTN, HLD, CAD s/p PCI with stent to LAD, mild sensorineural hearing loss who presents to establish care with GU Med Onc for newly diagnosed urethral carcinoma.  She had initially presented to her PCP with vaginal bleeding, an enlarging vaginal mass and several weeks of urinary frequency.  Today she reports that vaginal bleeding has stopped, however, she still has urinary frequency and now notes a weak urine stream.

Objective: Vitals signs: weight 91.1 kg, height 175 cm, BP 152/74. HR 76, RR 18, O2 saturation 95% on RA
Physical exam: General: alert, no acute distress, ECOG 1; Heart: RRR, normal S1/S2, no murmurs; Lungs: CTA b/l; Abd: Soft, nontender, non-distended; GU: On bi-manual exam there is a firm, fixed periurethral mass that appears to invade the anterior vagina; MSK: No LE edema; Skin: no rash; Neuro: No gross neurological deficits, CN 2-12 intact.
Labs significant for normal creatinine at 0.80 mg/dL.
CT urogram demonstrates a distal urethral lesion measuring 1.6 x 1.2 x 2.3 cm, but no evidence of pelvic lymphadenopathy or metastatic disease.
Pathology from core biopsy of periurethral mass reveals an invasive poorly differentiated carcinoma with squamous differentiation. IHC stains are positive for CK7, p63, p16, p40 and GATA3 and negative for CK20.\"
    """
sys_prompt_UC_init = \
    """
    You are a leading expert physician in Medical Oncology. Your goal is to formulate an Asessment & Plan (A&P), with the exact clinical stage diagnosis in TNM staging and the best definitive management for your patient's visit, using the provided Subjective and Objective note from your patient encounter. 
    Please keep in mind the patient encounter note may be missing pieces of information. If the patient encounter note contains [MISSING INFO], then you can assume that information is missing. For phrases where that is not present, missing information may or may not be present. Regardless, you are still responsible for fully understanding the note. Your patient encounter note (Subjective and Objective) is summarized below:
    """
sys_prompt_UC_end = \
    """
    You need to be very specific in your response, providing your rationale and using medical terminology where appropriate.
    """

user_prompt_UC = \
    """
    please only tell me:
    1. The best definitive management option for the patient if applicable
    
    Write "unknown" anywhere where the answer is not known.
    """
    # Based on your goal and this note,
    #     1.  Provide a detailed clinical assessment
    #     2.	Provide a detailed clinical plan based on your assessment
    #     3.	Identify the best option for definite management
    #     4.	Would you like to perform any additional tests? Only suggest additional tests if you think it is strictly necessary in significantly informing your plan.

    # Draft your response using the A&P style format note I have provided below. Strictly fill out the areas enclosed in “[]”.

    # A&P: [Patient Summary].  This appears to be a clinical stage [exact TNM clinical staging] [disease]. We discussed [definitive management], either as [definitive management options]. [Further details about definitive management] would be the preferred approach, except in [alternative disease], where [alternative disease definitive management]. Since her pathology shows [part(s) of patient clinic note that suggests most likely disease], I would favor [favored definitive management treatment]. [Consideration of additional tests if necessary based on patient’s subjective and objective]. [Explanation of patient’s symptoms]. [referrals if necessary].

    # please only tell me:
    # 1. The exact TNM clinical staging for the patient.
    # 2. The best definitive management option for the patient
    # 3. Would you like to perform any additional tests? Only suggest additional tests if you think it is strictly necessary in significantly informing your definitive management plan.
    



if PROMPT == "UC":
    inputs = [sys_prompt_UC_init, S_O_UC_var, sys_prompt_UC_end, user_prompt_UC]
elif PROMPT == "NSCLC-initiateTreatment":
    inputs = [sys_prompt_Squamous_Lung_init, S_O_Squamous_Lung_initiateTreatment, sys_prompt_Squamous_Lung_end, user_prompt_Squamous_Lung_initiateTreatment]
elif PROMPT == "NSCLC-RoutineFollowup":
    inputs = [sys_prompt_Squamous_Lung_init, S_O_Squamous_Lung_routineFollowup, sys_prompt_Squamous_Lung_end, user_prompt_Squamous_Lung_routineFollowup]
else:
    raise Exception("Please provide a prompt")


def extract_umls_concepts(p):
    """
    Extracts UMLS concepts from text using the scispaCy entity linker.
    """

    if p == "UC":
        terms = ["BASELINE","PMHx", "urethral carcinoma", "HTN", "HLD", "CAD s/p PCI with stent to LAD", "CAD", "PCI", "stent","LAD", "mild", "sensorineural hearing loss", \
        "sensorineural", "hearing", "loss", "hearing loss", "establish care with GU Med Onc", \
            "newly diagnosed urethral carcinoma", "enlarging", "vaginal bleeding", "enlarging vaginal mass", "vaginal mass", "mass", "bleeding", "vaginal", "urinary frequency", \
                "urinary", "frequency", "several weeks of urinary frequency", "several weeks", "vaginal bleeding has stopped", "stopped", "weak", "weak urine stream", "urine stream", \
                    "91.1 kg", "175 cm", "152/74", "76", "18", "95% on RA", "alert, no acute distress", "alert", "no acute distress", "ECOG 1", "RRR, normal S1/S2, no murmurs", \
                        "RRR","normal S1/S2", "no murmurs", "Soft", "nontender", "non-distended", "bi-manual exam", "firm", "fixed", "periurethral", "invade the anterior vagina", \
                            "anterior", "vagina", "CTA b/l", "Soft, nontender, non-distended", \
                                "On bi-manual exam there is a firm, fixed periurethral mass that appears to invade the anterior vagina", "neurological deficits", "deficits", \
                                    "CN 2-12 intact", "intact", "CN 2-12", "normal creatinine at 0.80 mg/dL", "normal", "creatinine", "distal", "urethral", "lesion", "urethral lesion",
                                        "periurethral mass", "invade", "anterior vagina", "no LE edema", "no rash", \
                                            "No gross neurological deficits, CN 2-12 intact", "normal creatinine", "0.80", "distal urethral lesion", "1.6 x 1.2 x 2.3", \
                                                "no evidence of pelvic lymphadenopathy", "no evidence of", "pelvic lymphadenopathy", "pelvic", "lymphadenopathy", "metastatic disease", "metastatic",\
                                                    "disease", "CT urogram", "Pathology from core biopsy", "invasive", \
                                                    "poorly differentiated", "poorly", "carcinoma", "squamous", "squamous differentiation", "invasive poorly differentiated carcinoma with squamous differentiation",\
                                                        "positive", "CK7", "p63", "p16", "p40", "GATA3", \
                                                            "negative", "CK20", "negative for CK20", "positive for CK7, p63, p16, p40 and GATA3"]
    elif p == "NSCLC-initiateTreatment":
        terms = ["BASELINE", "former heavy smoker", "newly diagnosed metastatic carcinoma", "metastatic carcinoma", "squamous cell carcinoma from pulmonary origin", "squamous cell carcinoma", \
        "pulmonary origin", "severe left flank pain", "spontaneous subcapsular renal hematoma", "left subcapsular hematoma measuring 2.5 cm", "left subcapsular hematoma", \
            "right renal cortical cysts measuring 2.5 cm", "right renal cortical cysts", "1.7 cm probable liver cyst", "liver cyst", "right 1.6 cm adrenal nodule", "adrenal nodule", "exophytic simple cyst in the right kidney",\
                "exophytic simple cyst", "heterogenous enhancing lesion in the left upper pole of left kidney with hematoma", "heterogenous enhancing lesion", "left upper pole", "left kidney with hematoma", \
                    "enhancing 2.3 cm lesion in the liver", "liver", "right upper lobe mass measuring 4.7 cm", "right upper lobe", "left kidney", "right renal", "left subcapsular", \
                        "mediastinal and hilar lymphadenopathy with scattered pulmonary nodules", "mediastinal", "hilar", "lymphadenopathy", "scattered pulmonary nodules", "pulmonary nodules",\
                            "liver biopsy", "poorly differentiated carcinoma", "99%", "60%", "CPS 60", "TP53, NOTCH2, and NF2", "TP53", "NOTCH2", "NF2", "PD-L1", "severe right flank pain",\
                                "bilateral subcapsular hematoma in both kidney", "bilateral", "both kidney", "AKI", "dialysis", "carcinoma of unknown primary", "significantly increased uptake in the left tonsils, left lung mass and thoracic lymph nodes, liver",\
                                    "significantly increased uptake", "left tonsils", "left lung mass", "thoracic lymph nodes", "liver", "left tonsils, left lung mass and thoracic lymph nodes, liver",\
                                        "few polyps (tubular adenomas and hyperplastic polyps)", "few polyps", "(tubular adenomas and hyperplastic polyps)", "negative for malignancy", "malignancy", "negative",\
                                            "right upper lobe lung mass", "Positive for non-small cell carcinoma, favoring squamous cell carcinoma (TPS 60%, tumor morphology is similar to that liver biopsy)",\
                                                "Positive for non-small cell carcinoma", "favoring squamous cell carcinoma", "(TPS 60%, tumor morphology is similar to that liver biopsy)", \
                                                    "Positive", "non-small cell carcinoma", "that liver biopsy", "46%", "pancreatic cancer", "46% probability being pancreatic cancer", "46% probability being pancreatic cancer and 11% pulmonary squamous cell carcinoma, and 11% probability from head and neck",\
                                                        "11% pulmonary squamous cell carcinoma", "11%", "pulmonary squamous cell carcinoma", "11% probability", "11% probability from head and neck", "head and neck", \
                                                            "ambulate without difficulty", "ambulate", "difficulty", "Pulmonary symptoms is stable, without significant worsening of chest pain, dyspnea, hemoptysis", \
                                                                "Pulmonary symptoms is stable", "significant worsening of chest pain, dyspnea, hemoptysis", "chest pain", "dyspnea", "hemoptysis", "urinary and still has indwelling foley catheter", \
                                                                    "urinary", "indwelling foley catheter", "any headache, vision changes, hearing changes, dysphagia, nausea, vomiting, abd pain, constipation, diarrhea, dysuria, focal weakness, leg swelling, peripheral neuropathy, fever, chill",\
                                                                        "denies", "headache", "vision changes", "hearing changes", "dysphagia", "nausea", "vomiting", "abd pain", "constipation", "diarrhea", "dysuria", "focal weakness", "leg swelling", "peripheral neuropathy", "fever", "chill", \
                                                                            "130/78",  "(BP Location: Left arm, BP Position: Sitting)", "70", "98.2 °F (36.8 °C) (Oral)", "18", "1.803 m (5\' 11\") Comment: reported by pt", \
                                                                                "100.2 kg (221 lb) Comment: with shoes on", "98%", "30.82 kg/m²", "Heavy Smoker", "status: 1", "male alert and oriented in no acute distress", \
                                                                                    "No scleral icterus. EOMI", "No oropharyngeal lesions or thrush", "no masses, symmetrical", "Normal effort. Clear to auscultation: A&P bilaterally, no crackles/rhonchi/wheezes", "Normal S1/S2, regular rate and rhythm, no murmurs, gallops, rubs. No peripheral edema bilaterally",\
                                                                                        "soft, non-tender. Bowel sounds normal", "no digital cyanosis or clubbing", "No rashes or lesions", "Cervical and supraclavicular nodes normal", "Cranial nerves grossly intact. No focal deficits","Affect appropriate for situation. Alert and oriented to person, place and time",\
                                                                                            "Non-small cell carcinoma, favor squamous cell carcinoma", "Poorly differentiated carcinoma", "4.3 CM right upper lobe mass. Scattered pulmonary nodules. Enlarged mediastinal lymph nodes", "Known progressing pulmonary metastases. Stable subcapsular renal hematoma bilaterally",\
                                                                                                "Significantly increased uptake in the left tonsils, left lung mass and thoracic lymph nodes, liver", "No metastatic disease"]

             
    elif p == "NSCLC-RoutineFollowup":
        terms = ["BASELINE", "discoid", "discoid lupus", "extensive", "smoking", "extensive smoking history", "history", "extensive smoking history (40 pack-years, quit 2019)", "(40 pack-years, quit 2019)",\
        "squamous cell carcinoma of the lung", "further management", "squamous cell carcinoma", "squamous", "squamous cell", "lung", "hemoptysis", "LLL completely atelectatic", "LLL", "atelectatic", "completely atelectatic",\
            "small left pleural effusion", "small left", "pleural effusion", "mildly enlarged AP node", "small bilateral adrenal lesions (1.5 cm on the right with hounsfield of 7, and 1.3 cm on the left with Hounsfield 20) concern for adenoma", \
                "small bilateral adrenal lesions", "(1.5 cm on the right with hounsfield of 7, and 1.3 cm on the left with Hounsfield 20)", "adenoma", "fungating bleeding mass", "left upper lobe", "distal left main bronchus", "Bronch", \
                    "60-70% obstruction to the airway", "no abnormal LN station were seen", "positive", "Biopsy positive for squamous cell carcinoma", "arterial embolization", "initial follow", "relatively well", "not had further hemoptysis", \
                        "significant chest pain", "intermittent mild cough", "feeling stable", "Interval History", "face and chest skin", "plaquenil", "denies", "headache, vision changes, hearing changes, dysphagia, nausea, vomiting, abd pain, constipation, diarrhea, dysuria, focal weakness, leg swelling, peripheral neuropathy, fever, chill", "headache", "vision changes", "hearing changes", "dysphagia", "nausea", "vomiting", "abd pain", "constipation", \
                            "diarrhea", "dysuria", "focal weakness", "leg swelling",  "peripheral neuropathy", "fever", "chill", "160/73 (BP Location: Right arm, BP Position: Sitting)", "78", "97.7 °F (36.5 °C) (Oral)", "14", "1.829 m (6')", \
                                 "127 kg (280 lb) Comment: with shoes and jacket on", "95%", "37.97 kg/m²", "Former", "status: 0", "male alert and oriented in no acute distress", "No scleral icterus. EOMI.", "No oropharyngeal lesions or thrush",\
                                    "no masses, symmetrical", "Normal effort. Clear to auscultation: A&P bilaterally, no crackles/rhonchi/wheezes", "Normal S1/S2, regular rate and rhythm, no murmurs, gallops, rubs. No peripheral edema bilaterally",\
                                        "soft, non-tender. Bowel sounds normal", "no digital cyanosis or clubbing", "No rashes or lesions", "Cervical and supraclavicular nodes normal", "Cranial nerves grossly intact. No focal deficits", "Affect appropriate for situation. Alert and oriented to person, place and time",\
                                            "left lower lobe",  "completely", "abrupt termination of the proximal left lower lobe bronchus", "obstructive from a mass", \
                                                "abrupt termination of the proximal left lower lobe bronchus concern for obstructive from a mass", "mildly enlarged mediastinal lymph node", "mediastinal lymph node", "mildly", "mildly enlarged", "enlarged",\
                                                    "lymph node", "Small bilateral adrenal lesion", "adrenal", "adrenal lesion", "bilateral", "Small bilateral", "None"]
    
    return terms
               
def remove_umls_terms(text, terms_to_remove):
    """
    Remove specified UMLS terms from text.
    
    Args:
        text (str): Input text
        terms_to_remove (list): List of terms to remove
    
    Returns:
        str: Modified text with specified terms removed
    """
    modified_text = text
    
    for term in terms_to_remove:
        # Create a case-insensitive pattern that matches the term as a whole word
        pattern = r'\b' + re.escape(term) + r'\b'
        modified_text = re.sub(pattern, '[MISSING INFO]', modified_text, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    modified_text = re.sub(r'\s+', ' ', modified_text)
    
    return modified_text.strip()


def generate_removal_combinations(umls_terms):
    """
    Generate different combinations of UMLS term removals.
    
    Args:
        umls_terms (list): List of UMLS terms
        max_combinations (int): Maximum number of combinations to generate
    
    Returns:
        list: List of tuples, each containing (terms_to_remove, description)
    """
    combinations = []
    
    # Single term removals
    for term in umls_terms:
        combinations.append(([term], f"removed_{term}"))
    
    return combinations


def save_removal_combinations(combinations, umls_concepts, prompt_type, max_combinations):
    """
    Save removal combinations to a file for reuse across models.
    
    Args:
        combinations (list): List of removal combinations
        umls_concepts (list): List of UMLS concepts
        prompt_type (str): Type of prompt (UC, NSCLC, etc.)
        max_combinations (int): Maximum number of combinations
    """
    combinations_data = {
        'combinations': combinations,
        'umls_concepts': umls_concepts,
        'prompt_type': prompt_type,
        'max_combinations': max_combinations,
        'generation_timestamp': str(datetime.datetime.now())
    }
    
    filename = f"data/{PROMPT}/removal_combinations_{prompt_type}_{max_combinations}.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'wb') as f:
        pickle.dump(combinations_data, f)
    
    print(f"Saved removal combinations to {filename}")


def load_removal_combinations(prompt_type, max_combinations):
    """
    Load removal combinations from file.
    
    Args:
        prompt_type (str): Type of prompt (UC, NSCLC, etc.)
        max_combinations (int): Maximum number of combinations
    
    Returns:
        tuple: (combinations, umls_concepts) or (None, None) if file doesn't exist
    """
    filename = f"data/{PROMPT}/removal_combinations_{prompt_type}_{max_combinations}.pkl"
    
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded existing removal combinations from {filename}")
        return data['combinations'], data['umls_concepts']
    else:
        print(f"No existing removal combinations found at {filename}")
        raise Exception("Generate removal combinations first.")


def get_or_create_removal_combinations(prompt_type, max_combinations):
    """
    Get existing removal combinations or create new ones.
    
    Args:
        umls_concepts (list): List of UMLS concepts
        prompt_type (str): Type of prompt (UC, NSCLC, etc.)
        max_combinations (int): Maximum number of combinations
        force_regenerate (bool): Force regeneration even if file exists
    
    Returns:
        tuple: (combinations, umls_concepts)
    """
    
    combinations, loaded_concepts = load_removal_combinations(prompt_type, max_combinations)
    assert combinations
    assert loaded_concepts
    return combinations, loaded_concepts
    
def normalize_formatting(text):
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Remove Markdown-style bolding (e.g., **Assessment:** → Assessment:)
        line = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", line)

        # If it starts with a number (e.g., "1. ", "12. "), preserve the number
        match = re.match(r"^(\d+\.)\s*(.*)", line)
        if match:
            number = match.group(1)
            content = match.group(2).strip()

            # Add sentence terminator if missing
            if content and not content.endswith(('.', '?', '!', ':', '"', "'")):
                content += '.'
            cleaned_lines.append(f"{number} {content}")
            continue

        # For bullets like "•", "-", "*", remove them but preserve sentence
        line = re.sub(r"^\s*[•\-*]+\s*", "", line)

        if line:
            if not line.endswith(('.', '?', '!', ':', '"', "'")):
                line += '.'
            cleaned_lines.append(line)

    # Collapse into a single paragraph
    text = ' '.join(cleaned_lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_clinical_note(raw_note):
    return normalize_formatting(raw_note)


USE_STRUCTURED_SCHEME = True  # flip to False to disable

#=================================#
# MODIFY SCHEME BASED ON TASK_TAG #
#=================================#
STRUCTURED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "guidelines_used": {"type": "array", "items": {"type": "string"}},
        "tnm": {"type": "string"},
        "clinical_stage": {"type": "string"},
        "definitive_management": {"type": "string"},
        "brief_justification": {"type": "string"}
    },
    "required": ["guidelines_used", "tnm", "clinical_stage", "definitive_management", "brief_justification"],
    "additionalProperties": False
}

SAFE_COT_SYSTEM_SUFFIX = f"""
You are a clinical decision-support assistant.
You may use a private scratchpad to reason, but NEVER reveal your notes.
Only output a single JSON object that strictly matches this JSON Schema:
{json.dumps(STRUCTURED_JSON_SCHEMA)}
Rules:
- If TNM cannot be determined from the provided info, set "tnm": "unknown".
- If you believe staging is pending completion, use "TBD"; otherwise use the exact TNM string.
- "clinical_stage" must reflect the TNM (e.g., "Stage IIIB").
- "brief_justification" is ≤ 2 sentences. Do NOT include step-by-step reasoning.
- List the guideline titles/versions in "guidelines_used" (e.g., "AJCC 8th ed.").
- Output JSON only. No prose or markdown.
"""

TASK_REMINDER = """
Fill the JSON fields by addressing, in order:
1) Which guidelines you used to determine staging and management.
2) Exact TNM (or "unknown"/"TBD" per the rules).
3) Resulting clinical stage.
4) Best definitive management option based on the above.
Return JSON only.
"""

# ---- Self-consistency for HF models (free-text responses) ----
import re, json
from collections import Counter

# ---- Minimal JSON parsing helpers (for structured output path) ----
def _extract_json_block(text: str) -> str | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else None

def _parse_json_obj(text: str):
    blob = _extract_json_block(text)
    if not blob:
        return None
    try:
        return json.loads(blob)
    except Exception:
        return None

# Self-consistency parameters
SELF_CONSISTENCY_N = 5
SELF_CONSISTENCY_TEMPERATURE = 0.6
SELF_CONSISTENCY_TOP_P = 0.9

def extract_management_from_freetext(response: str) -> str:
    """
    Extract the definitive management recommendation from a free-text response.
    Uses multiple regex patterns to find the key recommendation.
    Patterns are based on actual model output formats observed in experiments.
    """
    # Patterns to extract management recommendations (ordered by specificity)
    patterns = [
        # Direct patterns for "best definitive management" (Direct prompt format)
        r"best definitive management[^:]*:\s*([^\n]+)",
        r"definitive management[^:]*:\s*([^\n]+)",
        r"1\.\s*(?:the\s+)?best definitive management[^:]*:\s*([^\n]+)",
        
        # Full A&P format: Look for treatment in numbered items (especially 4, 5)
        r"(?:4|5)\.\s*(?:If no metastatic disease[^,]*,\s*will\s*)?(?:proceed with\s*)?([^\n]*(?:chemoradiation|chemotherapy|immunotherapy|surgery|radiation)[^\n]*)",
        r"(?:4|5)\.\s*([^\n]*(?:systemic therapy|definitive treatment|curative[- ]intent)[^\n]*)",
        
        # Anticipate next treatment options (from A&P template)
        r"Anticipate next treatment options?:\s*([^\n•]+)",
        r"next treatment options?[^:]*:\s*([^\n•]+)",
        
        # Current/recommended therapy patterns
        r"Current Therapy:\s*([^\n•]+)",
        r"(?:recommend|suggest|propose)[^:]*:\s*([^\n.]+(?:therapy|treatment|chemotherapy|immunotherapy)[^\n.]*)",
        r"treatment\s+(?:plan|option|recommendation)[^:]*:\s*([^\n.]+)",
        r"(?:first[- ]line|frontline)\s+(?:therapy|treatment)[^:]*:\s*([^\n.]+)",
        
        # Specific treatment modality patterns
        r"(?:proceed with|initiate|start)\s+([^\n.]*(?:chemoradiation|chemotherapy|immunotherapy|surgery)[^\n.]*)",
        r"(?:concurrent|definitive)\s+chemoradiation[^\n.]*",
        r"(?:systemic therapy)[^\n.]*(?:chemotherapy|immunotherapy|targeted therapy)[^\n.]*",
        
        # Specific drug/regimen patterns (for metastatic NSCLC)
        r"((?:pembrolizumab|nivolumab|atezolizumab|durvalumab|cemiplimab)[^\n.]*(?:plus|with|\+|\s+and\s+)[^\n.]*)",
        r"((?:carboplatin|cisplatin)[^\n.]*(?:plus|with|\+|\s+and\s+)[^\n.]*(?:paclitaxel|pemetrexed)[^\n.]*)",
        r"((?:carboplatin|cisplatin|paclitaxel|pemetrexed)[^\n.]*(?:plus|with|\+|\s+and\s+)[^\n.]*(?:pembrolizumab|immunotherapy)[^\n.]*)",
        
        # KEYNOTE study reference (often indicates immunotherapy recommendation)
        r"KEYNOTE[- ]?\d+[^\n.]*(?:pembrolizumab|immunotherapy)[^\n.]*",
        
        # Urethral carcinoma specific patterns
        r"(?:chemoradiation|trimodality therapy|bladder-sparing)[^\n.]*",
        r"(?:radical\s+)?(?:cystectomy|urethrectomy)[^\n.]*",
        
        # Workup/staging patterns (for incomplete staging)
        r"(?:complete\s+)?staging\s+workup[^\n.]*(?:PET|MRI|CT)[^\n.]*",
        r"(?:PET[/-]?CT|brain\s+MRI|molecular\s+testing)[^\n.]*staging[^\n.]*",
        
        # Fallback: look for numbered list item 1 with treatment keywords
        r"1[.\)]\s*([^\n]*(?:therapy|treatment|chemotherapy|immunotherapy|radiation|surgery|workup|staging)[^\n]*)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted = match.group(1).strip() if match.lastindex else match.group(0).strip()
            extracted = re.sub(r'\s+', ' ', extracted)
            extracted = extracted.rstrip('.,;:')
            extracted = re.sub(r'^(?:will\s+|proceed\s+with\s+)', '', extracted, flags=re.IGNORECASE)
            if len(extracted) > 10:
                return extracted
    return response[:200].strip()


def normalize_management_answer(answer: str) -> str:
    """
    Normalize a management answer for comparison during majority voting.
    Converts to lowercase, removes extra whitespace, and standardizes common terms.
    Maps to treatment categories used in analysis (Workup, Chemotherapy, Immunotherapy, etc.)
    """
    normalized = answer.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    
    if any(term in normalized for term in ['staging workup', 'pet/ct', 'pet-ct', 'pet ct', 
                                            'brain mri', 'molecular testing', 'ngs', 
                                            'complete staging', 'further staging']):
        return 'workup'
    
    if any(term in normalized for term in ['chemoimmunotherapy', 'chemo-immunotherapy',
                                            'chemotherapy plus pembrolizumab', 
                                            'pembrolizumab plus chemotherapy',
                                            'carboplatin paclitaxel pembrolizumab',
                                            'keynote-789', 'keynote-407', 'keynote-189']):
        return 'chemo-immunotherapy'
    
    if any(term in normalized for term in ['pembrolizumab monotherapy', 'single-agent pembrolizumab',
                                            'immunotherapy alone', 'checkpoint inhibitor']):
        return 'immunotherapy'
    
    if any(term in normalized for term in ['chemoradiation', 'chemo-radiation', 
                                            'concurrent chemoradiation', 'definitive chemoradiation',
                                            'concurrent chemotherapy and radiation']):
        return 'chemoradiation'
    
    if any(term in normalized for term in ['radiation therapy', 'radiotherapy', 'sbrt', 
                                            'stereotactic', 'external beam']):
        if 'chemo' not in normalized:
            return 'radiation'
    
    if any(term in normalized for term in ['surgery', 'surgical', 'resection', 'lobectomy',
                                            'pneumonectomy', 'cystectomy', 'urethrectomy']):
        return 'surgery'
    
    if any(term in normalized for term in ['chemotherapy', 'carboplatin', 'cisplatin', 
                                            'paclitaxel', 'pemetrexed', 'docetaxel']):
        if 'pembrolizumab' not in normalized and 'immunotherapy' not in normalized:
            return 'chemotherapy'
    
    if any(term in normalized for term in ['pembrolizumab', 'nivolumab', 'atezolizumab',
                                            'durvalumab', 'cemiplimab', 'immunotherapy',
                                            'pd-l1', 'pd-1', 'checkpoint']):
        if any(term in normalized for term in ['carboplatin', 'cisplatin', 'paclitaxel', 
                                                'pemetrexed', 'chemotherapy']):
            return 'chemo-immunotherapy'
        return 'immunotherapy'
    
    if any(term in normalized for term in ['targeted therapy', 'egfr', 'alk', 'ros1', 
                                            'braf', 'kras', 'osimertinib', 'erlotinib',
                                            'gefitinib', 'crizotinib', 'alectinib']):
        return 'targeted therapy'
    
    if any(term in normalized for term in ['trimodality', 'bladder-sparing', 'bladder sparing']):
        return 'trimodality therapy'
    
    if any(term in normalized for term in ['palliative', 'supportive care', 'best supportive',
                                            'hospice', 'comfort care']):
        return 'palliative care'
    
    replacements = [
        (r'pembrolizumab', 'pembrolizumab'),
        (r'pembro', 'pembrolizumab'),
        (r'keytruda', 'pembrolizumab'),
        (r'carboplatin\s*[+/and]\s*paclitaxel', 'carboplatin/paclitaxel'),
        (r'carbo\s*[+/]\s*taxol', 'carboplatin/paclitaxel'),
        (r'first[- ]line', 'first-line'),
        (r'front[- ]line', 'first-line'),
    ]
    
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    
    return normalized


def self_consistent_generate_hf(
    pipeline,
    messages,
    model_type: str,
    *,
    n: int = SELF_CONSISTENCY_N,
    **gen_kwargs,
) -> str:
    """
    Self-consistency generation for HuggingFace models.

    - If USE_STRUCTURED_SCHEME is True, we *attempt* to parse JSON outputs and majority-vote
      on the normalized `definitive_management` field.
    - If JSON parsing fails, we fall back to the free-text extraction+normalization
      used in `self_consistent_generate_hf_freetext`.
    """
    replies: list[str] = []
    parsed: list[dict] = []

    for _ in range(n):
        if model_type == "openbiollm":
            prompt = pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            out = pipeline(prompt, **gen_kwargs)
            reply = out[0]["generated_text"][len(prompt):]
        else:
            out = pipeline(messages, **gen_kwargs)
            gt = out[0]["generated_text"]
            if isinstance(gt, str):
                reply = gt
            else:
                reply = next(
                    (m.get("content", "") for m in reversed(gt) if m.get("role") in ("assistant", "model")),
                    gt[-1].get("content", ""),
                )
            reply = reply.split("<end_of_turn>")[0].strip()

        replies.append(reply)
        if USE_STRUCTURED_SCHEME:
            obj = _parse_json_obj(reply)
            if isinstance(obj, dict):
                parsed.append(obj)

    if parsed:
        buckets = [normalize_management_answer(o.get("definitive_management", "")) for o in parsed]
        votes = Counter(buckets)
        if votes:
            winner = max(votes, key=votes.get)
            for r in replies:
                o = _parse_json_obj(r)
                if isinstance(o, dict) and normalize_management_answer(o.get("definitive_management", "")) == winner:
                    return r
        return json.dumps(parsed[0], indent=2)

    extracted = [normalize_management_answer(extract_management_from_freetext(r)) for r in replies if r]
    if extracted:
        votes = Counter(extracted)
        winner = max(votes, key=votes.get)
        for r in replies:
            if normalize_management_answer(extract_management_from_freetext(r)) == winner:
                return r
    return replies[-1] if replies else ""


def self_consistent_generate_hf_freetext(pipeline, messages, model_type, n=SELF_CONSISTENCY_N, **gen_kwargs):
    """
    Sample n generations from HF model, extract management recommendations,
    and return the response with the majority-voted management answer.
    
    Args:
        pipeline: HuggingFace pipeline
        messages: List of message dicts with 'role' and 'content'
        model_type: 'openbiollm' or 'medgemma'
        n: Number of samples for self-consistency
        **gen_kwargs: Additional generation kwargs
    
    Returns:
        str: The full response text corresponding to the majority-voted management answer
    """
    responses = []
    extracted_managements = []
    
    for i in range(n):
        if model_type == "openbiollm":
            prompt = pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            out = pipeline(prompt, **gen_kwargs)
            reply = out[0]["generated_text"][len(prompt):]
        else:
            # Use messages for medgemma
            out = pipeline(messages, **gen_kwargs)
            gt = out[0]["generated_text"]
            if isinstance(gt, str):
                reply = gt
            else:
                reply = next(
                    (m.get("content", "") for m in reversed(gt) 
                     if m.get("role") in ("assistant", "model")),
                    gt[-1].get("content", "")
                )
            reply = reply.split("<end_of_turn>")[0].strip()
        
        responses.append(reply)
        
        management = extract_management_from_freetext(reply)
        normalized = normalize_management_answer(management)
        extracted_managements.append(normalized)
    
    if not responses:
        return ""
    
    votes = Counter(extracted_managements)
    
    if not votes:
        return responses[0]
    
    winner = max(votes, key=votes.get)
    
    for resp, mgmt in zip(responses, extracted_managements):
        if mgmt == winner:
            return resp
    
    return responses[0]



if __name__ == "__main__":
    os.makedirs(f"data/{PROMPT}/{MODEL}-{TASK_TAG}", exist_ok=True)
    
    if MODEL in ("gpt-5", "gpt-5-2025-08-07"):
        if not OPENAI_API_KEY and os.getenv("OPENAI_API_KEY") in (None, ""):
            print("Error: OPENAI_API_KEY not provided. Set env var or pass --openai_api_key.")
            exit(1)
        if OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        gpt5 = GPT5Model(GPT5Config(
            reasoning_effort=REASONING_EFFORT,
            text_verbosity=VERBOSITY,
            max_output_tokens=MAX_OUTPUT_TOKENS
        ))
        pipeline = None
        thinking_model = gpt5
        thinking_prompts = None
    elif MODEL == "openbiollm":
        model_id = "aaditya/OpenBioLLM-Llama3-70B"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
            },
        )
        thinking_model = None
        thinking_prompts = None
    elif MODEL == "medgemma":
        pipeline = transformers.pipeline(
            "text-generation",
            model="google/medgemma-27b-text-it",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        thinking_model = None
        thinking_prompts = None
    else:
        print(f"Unknown model: {MODEL}. Supported models: openbiollm, medgemma, gpt-5, gpt-5-2025-08-07")
        exit(1)

    umls_concepts = extract_umls_concepts(PROMPT)
    print(f"Found {len(umls_concepts)} UMLS concepts: {umls_concepts}")
    
    filename = f"data/{PROMPT}/removal_combinations_{PROMPT}_{len(umls_concepts)}.pkl"
    
    if not os.path.exists(filename):
        print("Generating new removal combinations...")
        combinations = generate_removal_combinations(umls_concepts)
        print(f"Generated {len(combinations)} removal combinations")
        save_removal_combinations(combinations, umls_concepts, PROMPT, len(umls_concepts))
        
    removal_combinations, loaded_umls = get_or_create_removal_combinations(PROMPT, len(umls_concepts))
    
    all_responses = []
    
    BATCHED_DONE = False
    
    if MODEL in ("openbiollm", "medgemma") and BATCH_SIZE > 1:
        pending = []
        for i, (terms_to_remove, description) in enumerate(removal_combinations):
            desc_processed = description.replace('/', '_').replace('%', '_').replace(':', '_').replace('.','_').replace(',', '_')
            filename = f"data/{PROMPT}/{MODEL}-{TASK_TAG}/routine_{PROMPT}_{MODEL}_{desc_processed}_2048tokens.txt"
            if os.path.exists(filename):
                with open(filename, "r") as model_response_exist:
                    file_list = model_response_exist.readlines()
                    response_e = "".join(file_list[3:])
                    all_responses.append({
                        'description': description,
                        'terms_removed': terms_to_remove,
                        'response': response_e
                    })
                continue
            pending.append((i, terms_to_remove, description))
        
        print(f"Batched generation enabled for {MODEL} with batch_size={BATCH_SIZE}. Pending items: {len(pending)}")
        
        for s in range(0, len(pending), BATCH_SIZE):
            batch = pending[s:s+BATCH_SIZE]
            print(f"\nProcessing batch {s//BATCH_SIZE + 1}/{(len(pending)+BATCH_SIZE-1)//BATCH_SIZE} (size={len(batch)})")
            batch_modified_inputs = []
            batch_desc_processed = []
            
            for (orig_idx, terms_to_remove, description) in batch:
                modified_system_prompt = remove_umls_terms(preprocess_clinical_note(inputs[1]), terms_to_remove)
                modified_inputs = [inputs[0], modified_system_prompt, *inputs[2:]]
                batch_modified_inputs.append(modified_inputs)
                batch_desc_processed.append(description.replace('/', '_').replace('%', '_').replace(':', '_').replace('.','_').replace(',', '_'))
            
            if MODEL == "openbiollm":
                terminators = [
                    pipeline.tokenizer.eos_token_id,
                    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                
                for j, mi in enumerate(batch_modified_inputs):
                    system_text = "".join([preprocess_clinical_note(x) for x in mi[:-1]])
                    user_text = preprocess_clinical_note(mi[-1])
                    if USE_STRUCTURED_SCHEME:
                        system_text = system_text + "\n" + SAFE_COT_SYSTEM_SUFFIX
                        user_text = user_text + "\n" + TASK_REMINDER
                    messages = [
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": user_text},
                    ]
                    
                    response = self_consistent_generate_hf(
                        pipeline,
                        messages,
                        model_type="openbiollm",
                        n=SELF_CONSISTENCY_N,
                        max_new_tokens=2048,
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=SELF_CONSISTENCY_TEMPERATURE,
                        top_p=SELF_CONSISTENCY_TOP_P,
                    )
                    
                    description = pending[s+j][2]
                    terms_to_remove = pending[s+j][1]
                    desc_processed = batch_desc_processed[j]
                    print(f"Response for {description}: {response[:200]}...")
                    filename = f"data/{PROMPT}/{MODEL}-{TASK_TAG}/routine_{PROMPT}_{MODEL}_{desc_processed}_2048tokens.txt"
                    with open(filename, "w") as file:
                        file.write(f"Terms removed: {terms_to_remove}\n")
                        file.write(f"Description: {description}\n")
                        file.write(f"Self-consistency: n={SELF_CONSISTENCY_N}, temp={SELF_CONSISTENCY_TEMPERATURE}\n")
                        file.write("-" * 50 + "\n")
                        file.write(response)
                    all_responses.append({
                        'description': description,
                        'terms_removed': terms_to_remove,
                        'response': response
                    })
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            elif MODEL == "medgemma":
                eos_ids = []
                try:
                    eot = pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")
                    if isinstance(eot, int) and eot >= 0:
                        eos_ids.append(eot)
                except Exception:
                    pass
                if pipeline.tokenizer.eos_token_id is not None:
                    eos_ids.append(pipeline.tokenizer.eos_token_id)
                
                for j, mi in enumerate(batch_modified_inputs):
                    system_text = "".join([preprocess_clinical_note(x) for x in mi[:-1]])
                    user_text = preprocess_clinical_note(mi[-1])
                    if USE_STRUCTURED_SCHEME:
                        system_text = system_text + "\n" + SAFE_COT_SYSTEM_SUFFIX
                        user_text = user_text + "\n" + TASK_REMINDER
                    messages = [
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": user_text},
                    ]
                    
                    response = self_consistent_generate_hf(
                        pipeline,
                        messages,
                        model_type="medgemma",
                        n=SELF_CONSISTENCY_N,
                        max_new_tokens=2048,
                        do_sample=True,
                        temperature=SELF_CONSISTENCY_TEMPERATURE,
                        top_p=SELF_CONSISTENCY_TOP_P,
                        eos_token_id=eos_ids or None,
                        return_full_text=False,
                        pad_token_id=pipeline.tokenizer.eos_token_id,
                    )
                    
                    description = pending[s+j][2]
                    terms_to_remove = pending[s+j][1]
                    desc_processed = batch_desc_processed[j]
                    print(f"Response for {description}: {response[:200]}...")
                    filename = f"data/{PROMPT}/{MODEL}-{TASK_TAG}/routine_{PROMPT}_{MODEL}_{desc_processed}_2048tokens.txt"
                    with open(filename, "w") as file:
                        file.write(f"Terms removed: {terms_to_remove}\n")
                        file.write(f"Description: {description}\n")
                        file.write(f"Self-consistency: n={SELF_CONSISTENCY_N}, temp={SELF_CONSISTENCY_TEMPERATURE}\n")
                        file.write("-" * 50 + "\n")
                        file.write(response)
                    all_responses.append({
                        'description': description,
                        'terms_removed': terms_to_remove,
                        'response': response
                    })
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        BATCHED_DONE = True

    if not BATCHED_DONE:
        for i, (terms_to_remove, description) in enumerate(removal_combinations):
            if terms_to_remove[0] == "any headache, vision changes, hearing changes, dysphagia, nausea, vomiting, abd pain, constipation, diarrhea, dysuria, focal weakness, leg swelling, peripheral neuropathy, fever, chill":
                description = "removed_any headache_ vision changes_ etc"
            desc_processed = description.replace('/', '_').replace('%', '_').replace(':', '_').replace('.','_').replace(',', '_')
            if desc_processed == "removed_headache_ vision changes_ hearing changes_ dysphagia_ nausea_ vomiting_ abd pain_                          constipation_ diarrhea_ dysuria_ focal weakness_ leg swelling_ peripheral neuropathy_ fever_ chill":
                desc_processed = "removed_headache_ vision changes_ etc"

            filename = f"data/{PROMPT}/{MODEL}-{TASK_TAG}/routine_{PROMPT}_{MODEL}_{desc_processed}_2048tokens.txt"
            if os.path.exists(filename):
                with open(filename, "r") as model_response_exist:
                    file_list = model_response_exist.readlines()
                    response_lines = file_list[3:]
                    if response_lines and response_lines[0].startswith("Generation time:"):
                        response_lines = response_lines[1:]
                    if response_lines and response_lines[0].strip().startswith("-" * 10):
                        response_lines = response_lines[1:]
                    response_e = "".join(response_lines)
                    all_responses.append({
                    'description': description,
                    'terms_removed': terms_to_remove,
                    'response': response_e
                    })
                continue
                
            print(f"\nTesting combination {i+1}/{len(removal_combinations)}: {description}")
            modified_system_prompt = remove_umls_terms(preprocess_clinical_note(inputs[1]), terms_to_remove)
            
            if MODEL in ("gpt-5", "gpt-5-2025-08-07"):
                system_content = "".join([
                    preprocess_clinical_note(inputs[0]),
                    preprocess_clinical_note(modified_system_prompt),
                    preprocess_clinical_note(inputs[2])
                ])
                thinking_result = thinking_model.generate_response(
                    system_prompt=system_content,
                    user_prompt=preprocess_clinical_note(inputs[-1])
                )
                
                if thinking_result.get("response"):
                    response = thinking_result["response"]
                    response_metadata = {
                        'generation_time': thinking_result.get("generation_time", 0),
                        'model_config': thinking_result.get("model_config", {})
                    }
                else:
                    print(f"Failed to generate GPT-5 response for {description}")
                    continue
                    
            else:
                modified_inputs = [inputs[0], modified_system_prompt, *inputs[2:]]

                system_text = "".join([preprocess_clinical_note(x) for x in modified_inputs[:-1]])
                user_text = preprocess_clinical_note(modified_inputs[-1])
                if USE_STRUCTURED_SCHEME:
                    system_text = system_text + "\n" + SAFE_COT_SYSTEM_SUFFIX
                    user_text = user_text + "\n" + TASK_REMINDER
                messages = [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ]
                
                if MODEL == "openbiollm":
                    terminators = [
                        pipeline.tokenizer.eos_token_id,
                        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    response = self_consistent_generate_hf(
                        pipeline,
                        messages,
                        model_type="openbiollm",
                        n=SELF_CONSISTENCY_N,
                        max_new_tokens=2048,
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=SELF_CONSISTENCY_TEMPERATURE,
                        top_p=SELF_CONSISTENCY_TOP_P,
                    )
                    response_metadata = {'self_consistency_n': SELF_CONSISTENCY_N}
                    
                elif MODEL == "medgemma":
                    eos_ids = []
                    try:
                        eot = pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")
                        if isinstance(eot, int) and eot >= 0:
                            eos_ids.append(eot)
                    except Exception:
                        pass
                    if pipeline.tokenizer.eos_token_id is not None:
                        eos_ids.append(pipeline.tokenizer.eos_token_id)

                    response = self_consistent_generate_hf(
                        pipeline,
                        messages,
                        model_type="medgemma",
                        n=SELF_CONSISTENCY_N,
                        max_new_tokens=2048,
                        do_sample=True,
                        temperature=SELF_CONSISTENCY_TEMPERATURE,
                        top_p=SELF_CONSISTENCY_TOP_P,
                        eos_token_id=eos_ids or None,
                        return_full_text=False,
                        pad_token_id=pipeline.tokenizer.eos_token_id,
                    )
                    response_metadata = {'self_consistency_n': SELF_CONSISTENCY_N}
            
            print(f"Response for {description}: {response[:200]}...")
            filename = f"data/{PROMPT}/{MODEL}-{TASK_TAG}/routine_{PROMPT}_{MODEL}_{desc_processed}_2048tokens.txt"
            with open(filename, "w") as file:
                    file.write(f"Terms removed: {terms_to_remove}\n")
                    file.write(f"Description: {description}\n")
                    if MODEL in ("gpt-5", "gpt-5-2025-08-07"):
                        file.write(f"Model: gpt-5\n")
                        file.write(f"Generation time: {response_metadata.get('generation_time',0):.2f}s\n")
                    file.write("-" * 50 + "\n")
                    file.write(response)
                    
            response_data = {
                'description': description,
                'terms_removed': terms_to_remove,
                'response': response
            }
            if MODEL in ("gpt-5", "gpt-5-2025-08-07"):
                response_data.update({
                    'generation_time': response_metadata.get("generation_time", 0),
                    'model_config': response_metadata.get("model_config", {})
                })
            all_responses.append(response_data)
            
            if MODEL not in ("gpt-5", "gpt-5-2025-08-07"):
                try:
                    del response, messages, modified_inputs
                except:
                    pass
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    try:
        summary_filename = f"data/{PROMPT}/{MODEL}-{TASK_TAG}/routine_{PROMPT}_{MODEL}_all_umls_removals_summary.txt"
        with open(summary_filename, "w") as file:
            file.write(f"UMLS Concepts Found: {loaded_umls}\n")
            file.write(f"Total combinations tested: {len(all_responses)}\n")
            if MODEL in ("gpt-5", "gpt-5-2025-08-07"):
                file.write(f"Reasoning effort: {REASONING_EFFORT}\n")
                file.write(f"Verbosity: {VERBOSITY}\n")
            else:
                file.write(f"Self-consistency: n={SELF_CONSISTENCY_N}, temperature={SELF_CONSISTENCY_TEMPERATURE}, top_p={SELF_CONSISTENCY_TOP_P}\n")
            file.write("=" * 80 + "\n\n")
            
            for response_data in all_responses:
                file.write(f"Combination: {response_data['description']}\n")
                file.write(f"Terms removed: {response_data['terms_removed']}\n")
                if MODEL in ("gpt-5", "gpt-5-2025-08-07") and response_data.get('generation_time'):
                    file.write(f"Generation time: {response_data['generation_time']:.2f}s\n")
                file.write("-" * 50 + "\n")
                file.write(response_data['response'])
        
        json_filename = f"data/{PROMPT}/{MODEL}-{TASK_TAG}/routine_{PROMPT}_{MODEL}_umls_removals_results.json"
        results_data = {
            'umls_concepts': loaded_umls,
            'total_combinations': len(all_responses),
            'model': MODEL,
            'prompt': PROMPT,
            'responses': all_responses
        }
        
        if MODEL in ("gpt-5", "gpt-5-2025-08-07"):
            results_data['thinking_model_config'] = {
                'reasoning_effort': REASONING_EFFORT,
                'verbosity': VERBOSITY
            }
        else:
            results_data['self_consistency_config'] = {
                'n_samples': SELF_CONSISTENCY_N,
                'temperature': SELF_CONSISTENCY_TEMPERATURE,
                'top_p': SELF_CONSISTENCY_TOP_P
            }
        
        with open(json_filename, "w") as file:
            json.dump(results_data, file, indent=2)
        
        print(f"All responses saved successfully. Total combinations tested: {len(all_responses)}")
        if MODEL in ("gpt-5", "gpt-5-2025-08-07"):
            print(f"GPT-5 thinking model used with reasoning_effort={REASONING_EFFORT}, verbosity={VERBOSITY}")
    except IOError as e:
        print(f"Error writing to file: {e}")