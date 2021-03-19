```python
import io
from nbformat import current
import os

def word_count_notebook(filename):
    with io.open(filename, 'r', encoding='utf-8') as f:
        nb = current.read(f, 'json')

    word_count = 0
    for cell in nb.worksheets[0].cells:
        if cell.cell_type == "markdown":
            word_count += len(cell['source'].replace('#', '').lstrip().split(' '))
            
    return word_count
```


```python
import re

pattern = re.compile(r'^[0-9].*') # if string being with number

total_word_count = 0
for filename in os.listdir("."):
    if re.match(pattern, filename):
        print(filename)
        wc = word_count_notebook(filename)
        print(wc)
        total_word_count+=wc
        
print(f"Total Word Count: {total_word_count}")
```

    0-introduction.ipynb
    147
    1-why-are-gas-fired-powerstations-important.ipynb
    895
    2-how-efficient-are-gas-fired-powerstations.ipynb
    0
    3-what-affects-efficiency.ipynb
    0
    Total Word Count: 1042
    


```python

```
