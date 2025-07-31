import pytest
import pokergym.agents.human as human_module


MASKS = [
    {
        "action": {
            "fold": True,
            "check": True,
            "call": True,
            "raise": True,
            "pass": True
        },
        "raise_amount": [10, 100]
    }
]
INPUTS = [
    "fold",
]
OUTPUTS = [
    {
        "action": "fold",
        "raise_amount": [0]
    }
]

class TestClass:

    @pytest.mark.parametrize("masks", MASKS)
    @pytest.mark.parametrize("inputs", INPUTS)
    @pytest.mark.parametrize("outputs", OUTPUTS)
    def test_function_1(self, masks, inputs, outputs):
        # Override the Python built-in input method 
        human_module.input = lambda: 'some_input'
        # Call the function you would like to test (which uses input)
        agent = human_module.HumanAgent(0)

        assert True
  

    def teardown_method(self, method):
        # This method is being called after each test case, and it will revert input back to original function
        human_module.input = input 

if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
