import xml.etree.ElementTree as ET
import xml.dom.minidom
import sys

def et(filepath, tag, txt = False):
    """
    Função para buscar elementos de uma tag específica.
    A função usa o método 'findall' que é case sensitive.
    :param str filepath: Caminho completo do XML que será lido.
    :param str tag: Tag para busca de elementos.
    :param bool txt: Saída sem tags de xml.
    :return list elements: Lista contendo todos os elementos encontrados.
    """

    tree = ET.parse(filepath)
    root = tree.getroot()

    elements_xml = root.findall(f".//{tag}")

    if txt:
        elements = [element.text for element in elements_xml]
    else:
        elements = [ET.tostring(element).decode("utf-8") for element in elements_xml]

    return elements



def dom(filepath, tag, txt=False):
    """
    Função para buscar elementos de uma tag específica.
    A função usa o método 'getElementsByTagName' que é case sensitive.
    :param str filepath: Caminho completo do XML que será lido.
    :param str tag: Tag para busca de elementos.
    :param bool txt: Saída sem tags de xml.
    :return list elements: Lista contendo todos os elementos encontrados.
    """
    doc = xml.dom.minidom.parse(filepath)

    elements_xml = doc.getElementsByTagName(f"{tag}")

    if txt:
        elements = [element.firstChild.nodeValue for element in elements_xml]
    else:
        elements = [element.toxml() for element in elements_xml]
    
    return elements


    


if __name__ == '__main__':

    authors = dom(sys.path[0]+'\\..\\data\\cf79.xml', 'AUTHOR', txt = False)
    with open(sys.path[0]+"\\..\\output\\autores.xml", "w") as f:
        f.writelines(author + '\n' for author in authors)


    titles = et(sys.path[0]+'\\..\\data\\cf79.xml', 'TITLE', txt = False)
    with open(sys.path[0]+"\\..\\output\\titulos.xml", "w") as f:
        f.writelines(title for title in titles)

