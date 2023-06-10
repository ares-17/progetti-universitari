# Utilizza l'immagine di base di pandoc/latex
FROM pandoc/latex

# Aggiungi un punto di montaggio per la cartella dei file LaTeX
VOLUME /src

# Esegui la conversione del file LaTeX in PDF utilizzando pandoc
CMD ["/src/main.tex", "-o", "/src/docs.pdf"]
