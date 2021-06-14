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

    C:\Users\rachel.hassall\.conda\envs\charterful\lib\site-packages\nbformat\current.py:15: UserWarning: nbformat.current is deprecated.
    
    - use nbformat for read/write/validate public API
    - use nbformat.vX directly to composing notebooks of a particular version
    
      warnings.warn("""nbformat.current is deprecated.
    


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
    153
    1-why-are-gas-fired-powerstations-important.ipynb
    1238
    2-how-efficient-are-gas-fired-powerstations.ipynb
    851
    3-what-affects-efficiency.ipynb
    243
    Total Word Count: 2485
    


```python

```
