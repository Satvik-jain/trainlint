def test_scaler_fitted_before_split(lint):
    findings = lint(
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
        """,
        "TL001",
    )
    assert [(f.rule, f.line) for f in findings] == [("TL001", 6)]
    assert "train_test_split" in findings[0].message


def test_feature_selection_before_cross_validation(lint):
    findings = lint(
        """
        X = SelectKBest(f_classif, k=20).fit_transform(X, y)
        scores = cross_val_score(model, X, y, cv=5)
        """,
        "TL001",
    )
    assert [f.line for f in findings] == [2]


def test_fit_then_transform_on_full_data(lint):
    findings = lint(
        """
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        X_train, X_test = train_test_split(X_scaled)
        """,
        "TL001",
    )
    assert [f.line for f in findings] == [2]


def test_chained_steps_are_each_reported(lint):
    findings = lint(
        """
        Xs = scaler.fit_transform(X)
        Xp = pca.fit_transform(Xs)
        a, b = train_test_split(Xp)
        """,
        "TL001",
    )
    assert [f.line for f in findings] == [2, 3]


def test_oversampling_before_split(lint):
    findings = lint(
        """
        X_res, y_res = SMOTE().fit_resample(X, y)
        a, b, c, d = train_test_split(X_res, y_res)
        """,
        "TL001",
    )
    assert len(findings) == 1


def test_split_first_is_clean(lint):
    assert not lint(
        """
        X_train, X_test = train_test_split(X)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        """,
        "TL001",
    )


def test_pipeline_is_clean(lint):
    assert not lint(
        """
        pipe = make_pipeline(StandardScaler(), SVC())
        scores = cross_val_score(pipe, X, y)
        """,
        "TL001",
    )


def test_label_encoding_before_split_is_clean(lint):
    assert not lint(
        """
        y_enc = LabelEncoder().fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc)
        """,
        "TL001",
    )


def test_estimator_fit_without_transform_is_clean(lint):
    assert not lint(
        """
        model.fit(X, y)
        a, b = train_test_split(X)
        """,
        "TL001",
    )


def test_fit_after_split_inside_cv_loop_is_clean(lint):
    assert not lint(
        """
        for train_idx, val_idx in KFold(5).split(X):
            scaler.fit(X[train_idx])
            Xv = scaler.transform(X[val_idx])
        """,
        "TL001",
    )


def test_fit_on_test_split(lint):
    findings = lint(
        """
        scaler.fit(X_test)
        vec.fit_transform(val_texts)
        """,
        "TL002",
    )
    assert [f.line for f in findings] == [2, 3]


def test_fit_on_train_and_transform_on_test_is_clean(lint):
    assert not lint(
        """
        scaler.fit(X_train)
        X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)
        interval = scaler.fit(interval_data)
        """,
        "TL002",
    )


def test_self_scaling_before_split(lint):
    findings = lint(
        """
        df = (df - df.mean()) / df.std()
        X_train, X_test = train_test_split(df)
        """,
        "TL013",
    )
    assert [(f.rule, f.line) for f in findings] == [("TL013", 2)]
    assert "train_test_split" in findings[0].message


def test_self_scaling_into_new_name_before_cv(lint):
    findings = lint(
        """
        Xn = X / X.max()
        scores = cross_val_score(model, Xn, y)
        """,
        "TL013",
    )
    assert [f.line for f in findings] == [2]


def test_scaling_with_split_first_is_clean(lint):
    assert not lint(
        """
        X_train, X_test = train_test_split(df)
        X_train = (X_train - X_train.mean()) / X_train.std()
        X_test = (X_test - X_train.mean()) / X_train.std()
        """,
        "TL013",
    )


def test_scaling_without_a_split_is_clean(lint):
    assert not lint(
        """
        df = (df - df.mean()) / df.std()
        model.fit(df, y)
        """,
        "TL013",
    )


def test_per_row_statistics_are_clean(lint):
    assert not lint(
        """
        df = df.sub(df.mean(axis=1), axis=0)
        scaled = (X - X.mean(axis=1)) / X.std(1)
        X_train, X_test = train_test_split(scaled)
        """,
        "TL013",
    )


def test_stats_of_another_name_are_clean(lint):
    assert not lint(
        """
        scaled = (values - reference.mean()) / reference.std()
        X_train, X_test = train_test_split(scaled)
        """,
        "TL013",
    )
