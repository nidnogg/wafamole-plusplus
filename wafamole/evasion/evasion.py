"""The main class of WAF-A-MoLE"""
import signal
import json

from multiprocessing import Pool

from wafamole.evasion.engine import CoreEngine
from wafamole.models import Model
from wafamole.payloadfuzzer.sqlfuzzer import SqlFuzzer
from wafamole.utils.check import type_check

map = Pool().map


class EvasionEngine(CoreEngine):
    """Evasion engine object.
    """

    def __init__(self, model: Model):
        """Initialize an evasion object.
        Arguments:
            model: the input model to evaluate

        Raises:
            TypeError: model is not Model
        """
        type_check(model, Model, "model")
        super(EvasionEngine, self).__init__(model)

    # def _mutation_round(self, payload, round_size):
    
    #     fuzzer = SqlFuzzer(payload)
    
    #     # Some mutations do not apply to some payloads
    #     # This removes duplicate payloads
    #     payloads = {fuzzer.fuzz() for _ in range(round_size)}
    #     results = map(self.model.classify, payloads)
    #     confidence, payload = min(zip(results, payloads))
    
    #     return confidence, payload

    def evaluate(
        self,
        payload: str,
        max_rounds: int = 1000,
        round_size: int = 20,
        timeout: int = 14400,
        threshold: float = 0.5,
    ):
        """It tries to produce a payloads that should be classified as a benign payload.

        Arguments:
            payload (str) : the initial payload
            max_rounds (int) : maximum number of mutation rounds
            round_size (int) : how many mutation for each round
            timeout (int) : number of seconds before the timeout
            threshold (float) : default 0.5, customizable for different results

        Raises:
            TypeError : input arguments are mistyped.

        Returns:
            float, str : minimum confidence and correspondent payload that achieve that score
        """

        type_check(payload, str, "payload")
        type_check(max_rounds, int, "max_rounds")
        type_check(round_size, int, "round_size")
        type_check(timeout, int, "timeout")
        type_check(threshold, float, "threshold")

        def _signal_handler(signum, frame):
            raise TimeoutError()

        # Timeout setup
        signal.signal(signal.SIGALRM, _signal_handler)
        signal.alarm(timeout)

        evaluation_results = []
        min_confidence, min_payload = self._mutation_round(payload, round_size)
        evaluation_results.append((min_confidence, min_payload))

        try:
            while max_rounds > 0 and min_confidence > threshold:
                for candidate_confidence, candidate_payload in sorted(
                    evaluation_results
                ):
                    max_rounds -= 1
            
                    confidence, payload = self._mutation_round(
                        candidate_payload, round_size
                    )
                    # print("Cur confidence {}\n Cur payload {}\n Rounds left {}\n".format(
                    #         confidence, payload, max_rounds
                    #         )
                    #     )
                    if confidence < candidate_confidence:
                        evaluation_results.append((confidence, payload))
                        min_confidence, min_payload = min(evaluation_results)
                        break

            if min_confidence < threshold:
                print("[+] Threshold reached")
                pass # comment if reusing print
            elif max_rounds <= 0:
                print("[!] Max number of iterations reached")
                # print("max_iter ")
                pass # comment if reusing print 

        except TimeoutError:
            print("[!] Execution timed out")
            # print("timeout")

        print(
            "Reached confidence {}\nwith payload\n{}".format(
                min_confidence, min_payload
            )
        )

        # Test print for mole_generator
        # print("{:.6} {}".format(min_confidence, max_rounds))
#       
        # outputJsonRow = {}
        # outputJsonRow["pattern"] = min_payload
        # outputJsonRow["type"] = "sqli"
        
        # # WARNING: This jsonOutputFilePath only works if wafamole is executed from datasets/mole_query_generator.py
        # jsonOutputFilePath = 'SQLiV3_mole.json'
        # with open(jsonOutputFilePath, 'a') as jsonFile:
        #     jsonFile.write(json.dumps(outputJsonRow, indent=4) + ',\n')

        return min_confidence, min_payload
