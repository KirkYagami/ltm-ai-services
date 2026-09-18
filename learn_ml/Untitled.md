```python
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression  
  
# Using f_regression  
selector = SelectKBest(score_func=f_regression, k=5)  
selector.fit(X, y)  
print(selector)  
  
  
# Using mutual_info_regression  
selector = SelectKBest(score_func=mutual_info_regression, k=5)  
selector.fit(X, y)  
print(selector)
```

