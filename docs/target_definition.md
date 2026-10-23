# Target definition

> **Status: NOT YET SETTLED.** Owner: Parth. Nothing downstream — split, features,
> models — should be finalised until this file is filled in, because the target
> determines whether the task is regression or classification and which columns must
> be dropped for leakage.

## 1. Target column

| Field | Value |
| :--- | :--- |
| Column name in the dataset | _TBD_ |
| Type | _continuous index / defined class scheme_ |
| Units or class labels | _TBD_ |
| Class balance or value range | _TBD_ |

## 2. Standard or index definition used

Name the single standard or index this project commits to, and cite its definition
(e.g. CCME WQI). Incompatible water quality standards must not be mixed.

_TBD_

## 3. Leakage assessment

If the target is an index computed from input columns, regressing it on those columns
is target leakage.

| Question | Answer |
| :--- | :--- |
| Is the target derived from any input column? | _TBD_ |
| Which columns contribute to it? | _TBD_ |
| Resolution chosen | _drop contributing inputs / choose a different target_ |

State the chosen resolution in the report's Methodology section, not only here.

## 4. Columns dropped as a result

_TBD — list each dropped column and the reason._
