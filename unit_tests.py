import unittest

from hybrid_text_classification import elaborate_prompt


class TestLLMClassificationOutputs(unittest.TestCase):

    def test_input_1(self):
        self.assertEqual(
            '{"selection": "CT"}',
            elaborate_prompt("Trovato apparato con versamento e prelievo fuori servizio, rimosso pezzi di banconote inceppate nel cash transport e consegnate al personale banca, sostituito trasporto banconote lower con cinghie usurate, carico atm e verifica funzionamento con operazioni clienti banca ")
            )

    def test_input_2(self):
        self.assertEqual(
            '{"selection": "CASSETTE"}',
            elaborate_prompt("Gr: ATM trovato in servizio. Trovate banconote accartocciate nel cassetto AC. Sostituito cassetto AC ed effettuate prove di dispensazione e versamento andate a buon fine. Rimesso in servizio ATM e fatte prove con cassiere positive. ")
            )

    def test_input_3(self):
        self.assertEqual(
            '{"selection": "NV"}',
            elaborate_prompt("ATM fuori servizio, nessuna banconota inceppata, errore 332f errore sui sensori dell'nv. Sostituito nv, eseguita pulizia analisi log atm bloccato da questo errore. Eseguite prove con clientela intesa sanpaolo con risultati positivi. ATM in regolare servizio ")
            )

    def test_input_4(self):
        self.assertEqual(
            '{"selection": "CASSETTE"}',
            elaborate_prompt("Sostituzione cassetti DC fit e unfit. Test ok")
            )

    def test_input_5(self):
        self.assertEqual(
            '{"selection": "CT"}',
            elaborate_prompt("GR anomalia 9516, le banconote si bloccano in dispensazione. Sostituiti piatti cash transport. Effettuata pulizia e prove di funzionamento con esito positivo. Rimesso in servizio l'apparato. ")
            )

    def test_input_6(self):
        self.assertEqual(
            '{"selection": "CT"}',
            elaborate_prompt("Ritrovato CT con cinghia rotta causa usura. Sostituito Ct. Test ok. ATM in servizio ")
            )

    def test_input_7(self):
        self.assertEqual(
             '{"selection": "NV"}',
             elaborate_prompt("sostituzione validatore banconote, reset e test ok")
        )

    def test_input_8(self):
        self.assertEqual(
            '{"selection": "NF"}',
            elaborate_prompt("Guasto. Banconote incastrate all'interno dello sfogliatore superiore. Soluzione. Rimosso inceppamento e reset dispositivo. Test di prelievo e versamento con esito positivo.")
            )

    def test_input_shutter(self):
        self.assertEqual(
            '{"selection": "SHUTTER"}',
            elaborate_prompt("Sostituzione monitor esterno e rimozione incpepamento nello shutter, reset e test ok")
            )

    def test_input_empty(self):
        self.assertEqual(
            '{"selection": "UNK"}',
            elaborate_prompt("")
            )

    def test_input_wrong(self):
        self.assertEqual(
            '{"selection": "UNK"}',
            elaborate_prompt("Ciao, sono Mario!")
            )

    def test_input_special(self):
        self.assertEqual(
            '{"selection": "UNK"}',
            elaborate_prompt("MTA in regolare servizio. Effettuato controllo visivo, e controllo LOG. Effettuate diverse prove di dispensazione e versamento tramite tool. Funzionamento ok.")
            )

if __name__ == '__main__':
    unittest.main()