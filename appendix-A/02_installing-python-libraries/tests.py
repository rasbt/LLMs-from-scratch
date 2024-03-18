from python_environment_check import main


def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert "FAIL" not in captured.out
