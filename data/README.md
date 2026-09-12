# Titanic dataset

`raw/titanic.csv` is the public Kaggle Titanic training dataset, mirrored by Data Science Dojo at
<https://github.com/datasciencedojo/datasets/blob/master/titanic.csv>.

`Survived` is the binary target: `0` means that the passenger did not survive and `1` means that
the passenger survived. The pipeline uses only `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and
`Embarked`; names, ticket identifiers, and cabin identifiers are not model inputs.
