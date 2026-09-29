# Representative Selection

[Overview](README.md) · [Numerical inventory](eligible_candidate_pool.csv)

## Method

Selection is a coverage-based comparison, not a claim of globally optimal gait quality. Each algorithm/displacement group retains a variable number of distinct checkpoints, spanning available displacement, maximum command switching, original-only versus added-penalty conditions, and command-hold duration. Smaller yaw and lateral drift are preferences, with no fixed hidden threshold. Zero observed falls and torso contact are required for published policy bundles.

Within each group, rows are ordered by increasing displacement for lookup. Displacement is not the objective to maximize. Similar values are retained only when other measured behavior differs materially. Selection was rechecked against all 177 numerical matches; the original 33 representatives are retained and 14 additional behaviors are included. The resulting 47 bundles comprise 33 target-band and 14 higher-displacement policies. This is not an assertion that every unselected video is visually inferior.

## Coverage and tradeoffs

| Algorithm | Target-band coverage | Reason for retained alternatives |
|---|---|---|
| DQN | 0.6716–0.7986; maxima 2/3/4; original-only and added; 10/30/100× | Lowest numerical match 0.6716 has 84.7° yaw and 1.031 BL drift. 0.7656 retains 0.34 s minimum hold. Nearby 0.7959/0.7986 trade 0.36/0.14 s minimum hold against 0.559/0.320 BL drift. No evaluated target-band point is below 0.6716. |
| DDQN | 0.6582–0.7998; maxima 2/3/4; original-only and added; 10/30/100× | 0.6194 is excluded for torso contact. 0.7786 provides 0.40 s minimum hold; 0.7884 provides lower drift but only 0.06 s minimum hold. Remaining points around 0.79 are not assumed worse, but would reduce displacement coverage. |
| PPO | 0.6173–0.7519; maxima 2/3/4; original-only and added; 5/10/30× | Lower-band and upper-band endpoints are retained with interior samples. 0.6550/0.6778 provide longer minimum holds than several numerically straighter alternatives; 0.7115 retains a short-pulse contrast. |
| Masked PPO | 0.6018–0.6816; maxima 2/3/4; original-only and added; 5/10/30× | Includes the lowest and highest eligible displacement. 0.6497 retains 0.28 s minimum hold; 0.6728 combines three switches and lower drift. No evaluated eligible point reaches 0.7. |
| SAC-D 1-step | 0.6101, 0.6356, 0.7744; maximum 2 | All three numerical matches are included. Their lateral drift is substantial (0.638–0.831 BL); inclusion does not certify straight locomotion. |
| SAC-D 5-step | No matches | 22 checkpoints have displacement in 0.6–0.8, but maximum switch counts are 14–38. They are not silently excluded by a yaw/drift threshold. |

Higher-displacement DQN representatives span 0.8158–1.0116 and DDQN representatives span 0.8310–1.0025. Seven from each are retained, including values above 1.0; the other algorithms have no numerical match above 0.8 under the shared switching criterion. These fourteen are comparisons, not preferred slow-gait recommendations.

## Additional coverage

The additional candidates provide lower-displacement alternatives, different switch counts, hold durations, penalty conditions and straightness tradeoffs. They are not claimed to be visually superior.

- **dqn-moderate-0.6716**: Lowest available target-band DQN displacement. Retained as a poor-straightness comparison: 84.7 degrees maximum yaw and 1.031 BL maximum drift. It is not a preferred straight-gait candidate.
- **dqn-moderate-0.7650**: Four-switch alternative near 0.7656: smaller maximum drift (0.288 versus 0.544 BL), but shorter minimum command hold (0.12 versus 0.34 s).
- **dqn-higher-0.9092**: Interior higher-displacement comparison with four maximum switches; fills the displacement interval between 0.8617 and 0.9486.
- **dqn-higher-0.9761**: Added-penalty four-switch comparison with 0.16 s minimum hold, compared with 0.06 s at 0.9983.
- **ddqn-moderate-0.7746**: Three-switch alternative with 0.22 s minimum hold; compared with the nearby two-switch 0.7786 candidate with 0.40 s minimum hold.
- **ddqn-moderate-0.7998**: Upper target-band two-switch comparison with 29.0 degrees maximum yaw and 0.34 s minimum hold. Inclusion does not imply that near-0.8 displacement is preferred.
- **ddqn-higher-0.8649**: Original-only two-switch comparison with 0.50 s median hold; fills the lower part of the higher-displacement group.
- **ddqn-higher-0.9796**: Two-switch higher-displacement alternative with 0.42 s median hold, distinct from the nearby four-switch 0.9937 candidate.
- **ppo-moderate-0.6234**: Lower-displacement added-penalty comparison: 0.243 BL maximum drift, but includes 0.02 s command pulses. Lower displacement alone does not establish slow limb transitions.
- **ppo-moderate-0.6459**: Lower-displacement original-only comparison with 27.7 degrees maximum yaw and 0.216 BL drift; includes 0.02 s command pulses.
- **ppo-moderate-0.6935**: Two-switch interior comparison with 0.10 s minimum and 0.50 s median command hold; provides another result below 0.70.
- **masked-ppo-moderate-0.6062**: Lower-end three-switch alternative with 0.10 s minimum hold; lateral drift reaches 0.559 BL.
- **masked-ppo-moderate-0.6081**: Lower-end two-switch alternative with 0.266 BL drift and 0.50 s median hold, but includes 0.02 s pulses.
- **masked-ppo-moderate-0.6495**: Straightness contrast to 0.6497: 0.194 versus 0.382 BL maximum drift, with 0.02 versus 0.28 s minimum hold. Similar displacement does not imply equivalent switching behavior.

## Video review and availability

The current video catalog contains all 65 target-band numerical matches and 24 higher-displacement comparisons. The GitHub-linked collection contains 47 policies (33 target-band + 14 higher-displacement); the additional collection contains 42 other candidates (32 target-band + 10 higher-displacement). These current sets do not overlap by checkpoint SHA-256. Additional high-displacement candidates cover five separated displacement intervals per algorithm and exclude the 14 packaged high-displacement models.

The shared upload interface cannot reorganize earlier uploads; those previous copies remain but are excluded from current catalog counts. See the [video catalog](VIDEO_CATALOG.md) for authoritative paths, direct playback links and transfer status. Additional videos preserve potentially useful unselected behavior without asserting that all their checkpoints are packaged or that numerical eligibility establishes hardware suitability.
