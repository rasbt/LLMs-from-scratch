# knowledgebase-parser README (partial)

Source: https://raw.githubusercontent.com/Flashpoint-Software/knowledgebase-parser/refs/heads/master/README.md

> Note: The fetcher returned a truncated excerpt. If you need the full file, provide a local copy or enable a full raw fetch method.

```
# Knowledgebase migration

## 1. (Optional) Cloning of the existing collection
-
Import the previously exported CSV of the original knowledgebase
- Convert
`section`, `stage`, `production stage` elements to the label type
- Figure out the
`listShortId`, and `uuids` of the `description` and `draft` fields.


...
# Knowledgebase migration
...
## 2.
Conversion
- Obtain an API-Key with write-access to the collection
- Edit the config
in `index.js`. In particular make sure that
	- The `baseUrl`, `apiToken`,
`targetListShortId` and `filesBaseUrl` are set corretly
	- Make sure `sourceDataPath`
or `sourceListShortId` are only set if you are converting a cloned list
		(This
is important because the `{uuid}_filesData` field will not be ex-/importd with
CSV cloning)
	- Make sure the `sourceKey` values are correct in the field
mappings
	- If you want to reuse existing fields as the targets of conversion, set
`targetKey` to an exiting field
		If you want to create a new field, set
`targetUuid: uuidv4()` and `targetName`

*Notes*
1) Items are requried to have unique
names.
...
# Knowledgebase migration
...
## 2.
...
2) If you are using a `.json` file for the source data (i.e. set
`sourceDataPath`), it expectes the following format:
```json
	{
		"kanbanData":
[
			{"usual": "kanban-data format"},
			"..."
		]
	}
```
...
# Knowledgebase migration
...
## 3. Execution
- Run the script
with `node index.html`
- It will first aquire all required data (source and
target data, which coincide if no sources (see 2.2.2) are provided)
- It will run
the text-fields through the specified converters. Some files might not be found,
if they are not present in the item. This is inconsistency in the source data-set
and can only be fixed manually. Note, that these might also be in the
draft-field and thus not public.
- It will update the target-list with the converted
content. To improve speed you can adjust the `maxConcurrency` config-value. When
exceeding 200 reqs/sec. you will run into API-limits at which point the script will
pause for 1 minute and retry.

...
# Knowledgebase migration
...
## 4. Setup the wiki-view
- Create a wiki-view
for the target-list
- Adjust the view-settings (especially set the correct
content element)
- Group by `section` or create a subitems hierarchy
# Knowledgebase migration

## 1. (Optional) Cloning of the existing collection
...
## 2.
...
## 3. Execution
...
## 4. Setup the wiki-view
...
```
