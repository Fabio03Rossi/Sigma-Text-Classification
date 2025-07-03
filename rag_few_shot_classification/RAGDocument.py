from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Usa il loader corretto in base al tipo di file
def get_loader(location):
    if location.endswith(".csv"):
        return CSVLoader(location)
    elif location.endswith(".xlsx"):
        return UnstructuredExcelLoader(location)
    return None

class RAGDocument:
    def __init__(self, rag_document):
        self.rag_document = rag_document
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            length_function=len
        )
        self.documents = []
        self.add_document(rag_document)

    def add_document(self, rag_document):
        # Carichiamo il dataset da interrogare in base alla colonna del file Excel (molteplici colonne o documenti possono essere aggiunti)
        loader = get_loader(rag_document)
        self.documents.extend(loader.load_and_split(self.text_splitter))

    def get_documents(self):
        return self.documents
