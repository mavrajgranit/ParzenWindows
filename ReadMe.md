# Parzen Windows

- Estimate a Gaussian Distribution
- Are able to estimate complex probability distributions
- Should be able to be used in Continuous Action Spaces in Reinforcement Learning
- Only need a few layers
- Can instantly get stuck depending on initialisation conditions

## Test Cases
- [X] One unnormalized input varying from small numbers(+,-) to rather large numbers(+,-)
    - Unable to learn multiple large numbers
    - This is a problem since we might want to predict vastly different values to take as actions(30, -45.5)
    - Using linear layers combined with a non-linearity seems to help
- [X] One normalized input(+,-)
    - Can't converge under large normalized values
    - Unstable
- [ ] Multiple unnormalized inputs varying from small numbers(+,-) to rather large numbers(+,-)
- [ ] Multiple normalized inputs numbers(+,-)