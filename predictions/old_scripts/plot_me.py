"""
plots all results .... not sure it works
"""

import ipywidgets as widgets
from ipywidgets import interactive
 
items = ['All']+sorted(df['Asset Type Description'].unique().tolist())
 
def view(x=''):
    if x=='All': return df
    return df[df['Asset Type Description']==x]
 
w = widgets.Select(options=items)
interactive(view, x=w)
