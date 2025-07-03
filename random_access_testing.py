import time

import numpy as np
import pandas as pd

from complete_evalutation import complete_evaluation, initialize

MODEL1 = "Jiva/xlm-roberta-large-it-mnli"
MODEL2 = "MoritzLaurer/bge-m3-zeroshot-v2.0"
MODEL3 = "models/models--tasksource--deberta-small-long-nli/snapshots/deberta-small-long-nli/"

def print_and_w(file, text):
    text = str(text)
    print(text)
    print(text, file=file)

def run_model(model, logname, area_confidence, action_confidence, tokenizer = None):
    count = 1
    initialize(model, tokenizer)
    with open(logname + ".txt", 'w') as f:
        print_and_w(f, "AREA_CONF: " + str(area_confidence))
        print_and_w(f, "ACTION_CONF: " + str(action_confidence))
        for row_name, row in df.iterrows():
            st1 = row['Closing Note']

            print_and_w(f, "\n-------------------------------")
            print_and_w(f, f"\n++{count}++\n")

            start_time = time.time()
            text, state_result, area_result, action_result = complete_evaluation(str(st1), area_confidence, action_confidence)
            end_time = time.time()

            print_and_w(f, text + "\n\n" + str(state_result) + "\n" + str(area_result) + "\n" + str(action_result))

            print_and_w(f, f"Time taken: {end_time - start_time:.2f} seconds")
            count += 1
            if count > 20:
                break

        print_and_w(f, "\n-------------------------------\n")

        text, state_result, area_result, action_result = complete_evaluation("""
        ATM fuori servizio. Errore 5306 permane dopo cambio sfogliatore inferiore per cinghia rotto. 
        Sostituto nf superiore, anomalia permane, sostituita scheda upper e anomalia permane, infine provato a sostituire la scheda lower e l'anomalia Ã© rimasta. 
        Trovato cavo connettore sfogliatore superiore tranciato. Recuperato cablaggio superiore, sostituito cablaggio, prove massive da tool con esito positivo. 
        ATM in servizio con saltuarie anomalie di lettura carta per problema giÃ  noto dell'host intesa sanpaolo. FunzionalitÃ  atm ripristinate.
        """, area_confidence, action_confidence)

        print_and_w(f, text + "\n\n" + str(state_result) + "\n" + str(area_result) + "\n" + str(action_result))


        print_and_w(f, "\n-------------------------------\n")

        text, state_result, area_result, action_result = complete_evaluation("""
        Gr versamento e prelievo fuori servizio causa cinghie disallineate..ripristinato allineamento in sede cinghie del note escrow. verifica di tutte le parti, sostituito n.1 piatto sul cash transport. Eseguite prove funzionamento con esito positivo
        """, area_confidence, action_confidence)

        print_and_w(f, text + "\n\n" + str(state_result) + "\n" + str(area_result) + "\n" + str(action_result))

        print_and_w(f, "\n-------------------------------\n")

        text, state_result, area_result, action_result = complete_evaluation("""
        ATM trovato con errore nei versamenti, trovato errore cassetto pieno a livello software, controllati cassetti e verificato funzionamento NV, nessuna anomalia riscontrata, cassetti non pieni, 
        eseguito controllo log riscontrato un errore per cassetto pieno in 5gg di funzionamento, riavviato PC ATM, eseguite diverse prove con esito positivo, ATM in regolare funzionamento.
        """, area_confidence, action_confidence)

        print_and_w(f, text + "\n\n" + str(state_result) + "\n" + str(area_result) + "\n" + str(action_result))

        print_and_w(f, "\n-------------------------------\n")

        text, state_result, area_result, action_result = complete_evaluation("""
        Apparato trovato spento, il direttore l'ha spento in modo brutale, compromettendo la configurazone del sistema, inoltre il monitor esterno non veniva rilevato. 
        Rieseguete le configurazioni e ripristino del monitor. 
        Una volta eseguito il reset anche sul dispensatore non sisono rilevati problemi. Test funzionali con esito positivo. Eseguiti prelievi e versamenti.
        """, area_confidence, action_confidence)

        print_and_w(f, text + "\n\n" + str(state_result) + "\n" + str(area_result) + "\n" + str(action_result))



df = pd.read_excel("docs/ReportFeedbacks_2025-05-12_11-25.xlsx", index_col=None, na_values=['NA'])

df.replace(np.nan, 0, inplace=True)

run_model(MODEL1, "xlm-roberta-large-it-mnli", 0.08, 0.12)
run_model(MODEL2, "bge-m3-zeroshot-v2.0", 0.2, 0.1)
run_model(MODEL3, "deberta-small-long-nli", 0.035, 0.11)