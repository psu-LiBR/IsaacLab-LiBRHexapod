# Representative Selection

[Overview](README.md) · [Numerical inventory](eligible_candidate_pool.csv)

## Method

Selection is a coverage-based comparison, not a claim of globally optimal gait quality. Each algorithm/displacement group retains up to five distinct checkpoints, spanning available displacement, maximum command switching, original-only versus added-penalty conditions, and command-hold duration. Smaller yaw and lateral drift are preferences, with no fixed hidden threshold. Zero observed falls and torso contact are required for published policy bundles.

Within each group, rows are ordered by increasing displacement for lookup. Displacement is not the objective to maximize. Similar values are retained only when other measured behavior differs materially. Selection was rechecked against all 177 numerical matches; the existing 33 representatives are retained because they provide distinct coverage and tradeoffs. This is not an assertion that every unselected video is visually inferior.

## Coverage and tradeoffs

| Algorithm | Target-band coverage | Reason for retained alternatives |
|---|---|---|
| DQN | 0.6839–0.7986; maxima 2/3/4; original-only and added; 10/30/100× | Lowest numerical match 0.6716 has 84.7° yaw and 1.031 BL drift. 0.7656 retains 0.34 s minimum hold. Nearby 0.7959/0.7986 trade 0.36/0.14 s minimum hold against 0.559/0.320 BL drift. No evaluated target-band point is below 0.6716. |
| DDQN | 0.6582–0.7884; maxima 2/3/4; original-only and added; 10/30/100× | 0.6194 is excluded for torso contact. 0.7786 provides 0.40 s minimum hold; 0.7884 provides lower drift but only 0.06 s minimum hold. Remaining points around 0.79 are not assumed worse, but would reduce displacement coverage. |
| PPO | 0.6173–0.7519; maxima 2/3/4; original-only and added; 5/10/30× | Lower-band and upper-band endpoints are retained with interior samples. 0.6550/0.6778 provide longer minimum holds than several numerically straighter alternatives; 0.7115 retains a short-pulse contrast. |
| Masked PPO | 0.6018–0.6816; maxima 2/3/4; original-only and added; 5/10/30× | Includes the lowest and highest eligible displacement. 0.6497 retains 0.28 s minimum hold; 0.6728 combines three switches and lower drift. No evaluated eligible point reaches 0.7. |
| SAC-D 1-step | 0.6101, 0.6356, 0.7744; maximum 2 | All three numerical matches are included. Their lateral drift is substantial (0.638–0.831 BL); inclusion does not certify straight locomotion. |
| SAC-D 5-step | No matches | 22 checkpoints have displacement in 0.6–0.8, but maximum switch counts are 14–38. They are not silently excluded by a yaw/drift threshold. |

Higher-displacement DQN representatives span 0.8158–1.0116 and DDQN representatives span 0.8310–1.0025. Five from each are retained, including values above 1.0; the other algorithms have no numerical match above 0.8 under the shared switching criterion. These ten are comparisons, not preferred slow-gait recommendations.

## Video review and availability

The video delivery covers **all 65 target-band numerical matches**, plus only the **10 published higher-displacement comparisons**. The 33 videos for packaged policies are in `github-policy-videos`; the other 42 target-band videos are in `additional-simulation-videos`. The 112 higher-displacement population is documented without uploading all 112 videos. Historical and below-band videos are outside this delivery.

The additional collection preserves potentially useful unselected behaviors for visual review. Its checkpoints are not packaged in this GitHub delivery. The checkpoint hash identifies the exact source model if a further candidate is selected for packaging. Numeric coverage and qualitative appearance are different evidence; neither establishes physical-robot performance.
