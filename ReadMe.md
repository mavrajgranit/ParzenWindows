# Parzen Windows

- Estimate a Gaussian Distribution
- Are able to estimate complex probability distributions
- Should be able to be used in Continuous Action Spaces in Reinforcement Learning
- Only need a few layers
- Can instantly get stuck depending on initialisation conditions

## Test Cases
- [X] One unnormalized input varying from small numbers(+,-) to rather large numbers(+,-)
    - Unable to learn multiple large numbers
- [ ] One normalized input(+,-)
- [ ] Multiple unnormalized inputs varying from small numbers(+,-) to rather large numbers(+,-)
- [ ] Multiple normalized inputs numbers(+,-)