import pweave

pweave.weave(file = "./document.py", informat = "script", doctype = "markdown", output = "./document.md") 
# pweave.publish(file = "./document.py", doc_format = "pdf", output = "./document.pdf") 