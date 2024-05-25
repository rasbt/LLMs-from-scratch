# Chapter 7: Instruction and Preference Finetuning

This folder contains utility code that can be used for preparing an instruction dataset.



### Finding near duplicates

The `find-near-duplicates.py` function can be used to identify duplicates and near-duplicates in an instruction dataset. For example,



```python
python find-near-duplicates.py --json_file instruction-examples.json
```

```


==================================================
 Searching 'instruction' for duplicates ...
==================================================
Duplicate pair found with similarity 0.85:
1. Determine the state of matter for helium at room temperature.
2. Determine the state of matter for nitrogen at room temperature.

Duplicate pair found with similarity 0.98:
1. Edit the following sentence to make it more formal.
2. Edit the sentence to make it more formal.

Duplicate pair found with similarity 1.00:
1. Name a dwarf planet in our solar system.
2. Name a dwarf planet in our solar system.

Duplicate pair found with similarity 0.88:
1. Change the sentences from active voice to passive voice.
2. Change the sentence from passive to active voice.



==================================================
 Searching 'input' for duplicates ...
==================================================
Duplicate pair found with similarity 0.88:
1. 
2. She said, "I am tired."



==================================================
 Searching 'output' for duplicates ...
==================================================
Duplicate pair found with similarity 0.82:
1. Helium is in a gaseous state at room temperature.
2. Nitrogen is in a gaseous state at room temperature.

Duplicate pair found with similarity 1.00:
1. One dwarf planet in our solar system is Pluto.
2. One dwarf planet in our solar system is Pluto.


```

