# Third-Party Notices

Last Updated: 2026-01-18

This file contains notices for third-party software components included in or
derived from this project.

---

## LTX-2 (Lightricks)

**Components using LTX-2 code:**
- `src/llm_dit/schedulers/ltx2_scheduler.py` - Diffusion schedulers
- `src/llm_dit/models/ltx2/vae/` - Video VAE encoder/decoder
- `src/llm_dit/models/ltx2/transformer.py` - Diffusion transformer architecture
- `src/llm_dit/models/ltx2/connectors.py` - Text conditioning connectors
- `src/llm_dit/models/ltx2/attention.py` - Attention mechanisms
- `src/llm_dit/models/ltx2/components.py` - Core building blocks
- `src/llm_dit/models/ltx2/rope.py` - Rotary position embeddings
- `src/llm_dit/pipelines/generate.py` - Generation loop (algorithm design)

**Original Source:** https://github.com/Lightricks/LTX-2

**License:** LTX-2 Community License

**Copyright:** Copyright (c) 2025 Lightricks Ltd.

**Modifications:**
- Removed dependencies on internal Lightricks packages (ltx_core.*)
- Added Python type hints and docstrings
- Restructured into modular package layout
- Added support for multiple attention backends (FA3, xFormers, SDPA)
- Made scipy dependency optional for BetaScheduler
- Integrated with llm-dit project patterns

**License Text:**

```
LTX-2 COMMUNITY LICENSE AGREEMENT

Version Release Date: January 6, 2025

"Agreement" means the terms and conditions for use, reproduction, distribution,
and modification of the LTX-2 Materials set forth herein.

"Documentation" means the specifications, manuals, and documentation
accompanying LTX-2 distributed by Lightricks.

"Licensee" or "you" means you, or your employer or any other person or entity
(if you are entering into this Agreement on such person or entity's behalf),
of the age required under applicable laws, rules, or regulations to provide
legal consent, and that has legal authority to bind your employer or such other
person or entity if you are entering in this Agreement on their behalf.

"LTX-2" means the foundational video generation models and software and
algorithms, including machine-learning model code, trained model weights,
inference-enabling code, training-enabling code, fine-tuning-enabling code,
and other elements of the foregoing distributed by Lightricks.

"LTX-2 Materials" means, collectively, Lightricks's proprietary LTX-2 and
Documentation (and any portion thereof) made available under this Agreement.

"Lightricks" or "we" means Lightricks Ltd.

By using, reproducing, modifying, distributing, performing, or displaying any
portion or element of the LTX-2 Materials, or otherwise accepting the terms and
conditions of this Agreement, you agree to be bound by this Agreement.

1. License Rights and Redistribution

  a. Grant of Rights. Subject to the terms and conditions of this Agreement,
     Lightricks grants to you, free of charge and royalty-free, a non-exclusive,
     worldwide, non-transferable, and revocable license to use, reproduce,
     distribute, create derivative works of, and make modifications to the
     LTX-2 Materials.

  b. Redistribution and Use.

     i. You may make modifications to the LTX-2 Materials, including creating
        derivative works, provided you meet the following conditions:
        * You must provide a copy of this Agreement with every copy of the
          LTX-2 Materials, or works derived from the LTX-2 Materials (whether
          as standalone products, software, or in any other form), that you
          distribute, or otherwise make available to any person or entity
          (each, a "Recipient"), and each Recipient must agree to its terms
          and conditions.
        * You must give any Recipient a prominent notice stating that you
          modified the LTX-2 Materials.
        * You must retain in all copies of the LTX-2 Materials that you
          distribute the following attribution notice within a "Notice" text
          file distributed as a part of such copies: "LTX-2 is licensed under
          the LTX-2 Community License Agreement, Copyright (c) Lightricks. All
          Rights Reserved."

     ii. Your use of the LTX-2 Materials must comply with applicable laws and
         regulations (including trade compliance laws and regulations), and you
         must not use the LTX-2 Materials for any purpose prohibited by this
         Agreement or applicable laws or regulations.

2. Additional Commercial Terms

   If you or your affiliates are offering a commercial product or service using
   the LTX-2 Materials, you must contact Lightricks via license@lightricks.com
   to request permission and grant of a commercial license.

3. Disclaimer of Warranty

   UNLESS REQUIRED BY APPLICABLE LAW, THE LTX-2 MATERIALS ARE PROVIDED ON AN
   "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
   INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT,
   MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY
   RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING
   THE LTX-2 MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE
   LTX-2 MATERIALS AND ANY DERIVATIVE WORKS THEREOF OR OUTPUT GENERATED
   THEREFROM.

4. Limitation of Liability

   IN NO EVENT WILL LIGHTRICKS OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCT LIABILITY, OR
   OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY
   INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY, OR PUNITIVE DAMAGES,
   EVEN IF LIGHTRICKS OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF
   ANY OF THE FOREGOING.

5. Intellectual Property

   a. No trademark licenses are granted under this Agreement, and in connection
      with the LTX-2 Materials, neither Lightricks nor Licensee may use any name
      or mark owned by or associated with the other or any of its affiliates,
      except as required for compliance with the notice requirement of this
      Agreement, or as required for reasonable and customary use in describing
      and redistributing the LTX-2 Materials.

   b. Subject to Lightricks's ownership of LTX-2 Materials and the derivative
      works made by or for Lightricks, with respect to any derivative works and
      modifications of the LTX-2 Materials that you make, as between you and
      Lightricks, you are and will be the owner of such derivative works and
      modifications.

   c. If you institute litigation or other proceedings against Lightricks or any
      entity (including a cross-claim or counterclaim in a lawsuit) alleging
      that the LTX-2 Materials or any output thereof, or any part of either of
      the foregoing, constitutes infringement of intellectual property or other
      rights owned or licensable by you, then any licenses granted to you under
      this Agreement shall terminate as of the date such litigation or claim is
      filed or instituted. Lightricks shall have the right to terminate this
      Agreement if you are in breach of any term or condition of this Agreement.
      Upon termination of this Agreement, you shall delete and cease use of the
      LTX-2 Materials.

6. Term and Termination

   The term of this Agreement will commence upon your acceptance of this
   Agreement and will continue in full force and effect until terminated in
   accordance with the terms and conditions herein. Lightricks may terminate
   this Agreement if you are in breach of any term or condition of this
   Agreement. Upon termination of this Agreement, you shall delete and cease
   use of the LTX-2 Materials. Sections 3, 4, and 6 shall survive the
   termination of this Agreement.

7. Governing Law

   This Agreement shall be governed by and construed in accordance with the
   internal laws of the State of Israel without giving effect to any choice
   or conflict of law provision or rule that would require or permit the
   application of the laws of any jurisdiction other than those of the State
   of Israel.
```

---

## Other Attributions

Additional third-party components may have their own license files included
in their respective directories or documented in code comments.
