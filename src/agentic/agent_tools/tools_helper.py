from docling.document_converter import DocumentConverter
from src.utils.logger import MainLogger

def extract_data_dictionary(url: str):
    logger = MainLogger(__name__)
    tables = DocumentConverter().convert(url).document.export_to_dict().get('tables', [])

    colunas, caracteristicas, tipos = [], [], []
    tipos_colunas = ['varchar', 'date', 'number']

    for table in tables:
        for conteudo in table['data']['table_cells']:
            texto = conteudo.get('text', '')
            if texto in ['EVOLUCAO', 'UTI', 'DT_NOTIFIC', 'SEM_NOT', 'SG_UF_NOT', 'VACINA_COV', 'HOSPITAL', 'SEM_NOT']:
                colunas.append(texto)
            elif tipos_colunas[0] in texto.lower() or tipos_colunas[1] in texto.lower() or tipos_colunas[2] in texto.lower():
                tipos.append(texto)
            elif 'campo' in texto.lower():
                caracteristicas.append(texto)
        logger.info(f"{texto}")

    structs = {}
    for i in range(len(colunas)):
        structs[colunas[i]] = {
            'tipo': tipos[i] if i < len(tipos) else None,
            'caracteristicas': caracteristicas[i] if i < len(caracteristicas) else None
        }

    logger.info(f"{structs}")
    return structs