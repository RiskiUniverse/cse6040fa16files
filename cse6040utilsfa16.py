#!/usr/bin/env python3
"""
cse6040utilsfa16.py: Utility functions for CSE 6040, Fall 2016 edition
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

# ========== Helper function from Lab 5 ==========
def peek_table (db, name, num=5):
    """
    Given a database connection (`db`), prints both the number of
    records in the table as well as its first few entries.
    """
    count = '''select count (*) FROM {table}'''.format (table=name)
    peek = '''select * from {table} limit {limit}'''.format (table=name, limit=num)

    print ("Total number of records:", pd.read_sql_query (count, db)['count (*)'].iloc[0], "\n")

    print ("First {} entries:".format (num))
    display (pd.read_sql_query (peek, db))

def list_tables (conn):
    """Return the names of all visible tables, given a database connection."""
    query = """select name from sqlite_master where type = 'table';"""
    c = conn.cursor ()
    c.execute (query)
    table_names = [t[0] for t in c.fetchall ()]
    return table_names

# ========== Helper functions from Lab 8 ==========
def canonicalize_tibble (X):
    """Returns a tibble in _canonical order_."""
    # Enforce Property 1:
    var_names = sorted (X.columns)
    Y = X[var_names].copy ()

    # Enforce Property 2:
    Y.sort_values (by=var_names, inplace=True)

    # Enforce Property 3:
    Y.set_index ([list (range (0, len (Y)))], inplace=True)

    return Y

def tibbles_are_equivalent (A, B):
    """Given two tidy tables ('tibbles'), returns True iff they are
    equivalent.
    """
    A_canonical = canonicalize_tibble (A)
    B_canonical = canonicalize_tibble (B)
    cmp = A_canonical.eq (B_canonical)
    return cmp.all ().all ()


# ========== Helper functions from Lab 10 ==========
def make_scatter_plot (df, x="x_1", y="x_2", hue="label",
                       palette={0: "red", 1: "olive", 2: "blue", 3: "green"},
                       size=5,
                       centers=None):
    if (hue is not None) and (hue in df.columns):
        sns.lmplot (x=x, y=y, hue=hue, data=df, palette=palette,
                    fit_reg=False)
    else:
        sns.lmplot (x=x, y=y, data=df, fit_reg=False)

    if centers is not None:
        plt.scatter (centers[:,0], centers[:,1],
                     marker=u'*', s=500,
                     c=[palette[0], palette[1]])

def make_scatter_plot2 (df, x="x_1", y="x_2", hue="label", size=5):
    if (hue is not None) and (hue in df.columns):
        sns.lmplot (x=x, y=y, hue=hue, data=df,
                    fit_reg=False)
    else:
        sns.lmplot (x=x, y=y, data=df, fit_reg=False)

def mark_matches (a, b, exact=False):
    """
    Given two Numpy arrays of {0, 1} labels, returns a new boolean
    array indicating at which locations the input arrays have the
    same label (i.e., the corresponding entry is True).

    This function can consider "inexact" matches. That is, if `exact`
    is False, then the function will assume the {0, 1} labels may be
    regarded as the same up to a swapping of the labels. This feature
    allows

      a == [0, 0, 1, 1, 0, 1, 1]
      b == [1, 1, 0, 0, 1, 0, 0]

    to be regarded as equal. (That is, use `exact=False` when you
    only care about "relative" labeling.)
    """
    assert a.shape == b.shape
    a_int = a.astype (dtype=int)
    b_int = b.astype (dtype=int)
    all_axes = tuple (range (len (a.shape)))
    assert ((a_int == 0) | (a_int == 1)).all ()
    assert ((b_int == 0) | (b_int == 1)).all ()

    exact_matches = (a_int == b_int)
    if exact:
        return exact_matches

    assert exact == False
    num_exact_matches = np.sum (exact_matches)
    if (2*num_exact_matches) >= np.prod (a.shape):
        return exact_matches
    return exact_matches == False # Invert

def count_matches (a, b, exact=False):
    """
    Given two sets of {0, 1} labels, returns the number of mismatches.

    This function can consider "inexact" matches. That is, if `exact`
    is False, then the function will assume the {0, 1} labels may be
    regarded as similar up to a swapping of the labels. This feature
    allows

      a == [0, 0, 1, 1, 0, 1, 1]
      b == [1, 1, 0, 0, 1, 0, 0]
 
    to be regarded as equal. (That is, use `exact=False` when you
    only care about "relative" labeling.)
    """
    matches = mark_matches (a, b, exact=exact)
    return np.sum (matches)

# Adapted from: https://blog.dominodatalab.com/topology-and-density-based-clustering/
def make_crater (inner_rad=4, outer_rad=4.5, donut_len=2, inner_pts=1000, outer_pts=500, label=False):
    # Make the inner core
    radius_core = inner_rad*np.random.random (inner_pts)
    direction_core = 2*np.pi*np.random.random (size=inner_pts)

    # Simulate inner core points
    core_x = radius_core*np.cos (direction_core)
    core_y = radius_core*np.sin (direction_core)
    crater_core = pd.DataFrame ({'x_1': core_x, 'x_2': core_y})
    if label: crater_core['label'] = 0

    # Make the outer ring
    radius_ring = outer_rad + donut_len*np.random.random(outer_pts)
    direction_ring = 2*np.pi*np.random.random(size = outer_pts)

    # Simulate ring points
    ring_x = radius_ring*np.cos(direction_ring)
    ring_y = radius_ring*np.sin(direction_ring)
    crater_ring = pd.DataFrame ({'x_1': ring_x, 'x_2': ring_y})
    if label: crater_ring['label'] = 1

    return pd.concat ([crater_core, crater_ring])

# ========== Helper functions from Lab 12 ==========
from PIL import Image

def load_image_file (fn):
    return Image.open (fn, 'r')

def im2gnp (image):
    """Converts a PIL image into an image stored as a 2-D Numpy array in grayscale."""
    return np.array (image.convert ('L'))

def gnp2im (image_np):
    """Converts an image stored as a 2-D grayscale Numpy array into a PIL image."""
    return Image.fromarray (image_np.astype (np.uint8), mode='L')

def imshow_gray (im, ax=None):
    if ax is None:
        f = plt.figure ()
        ax = plt.axes ()
    ax.imshow (im,
               interpolation='nearest',
               cmap=plt.get_cmap ('gray'))

# eof
