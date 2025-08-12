import {
  Streamlit,
  StreamlitComponentBase,
  with</Streamlit>
} from "streamlit-component-lib";
import React, { ReactNode } from "react";
import "katex/dist/katex.min.css"; // For basic math rendering
import "tikzjax/dist/tikzjax.min.css"; // For TikZ diagrams
import "tikzjax/dist/tikzjax.min.js"; // TikZ-cd renderer

class LatexRendererComponent extends StreamlitComponentBase {
  public render = (): ReactNode => {
    const code = this.props.args["code"];

    return (
      <div>
        <h4>Rendered Diagram</h4>
        <div className="tikz-cd-container">
          <script type="text/tikz">
            {code}
          </script>
        </div>
      </div>
    );
  };
}

export default withStreamlitAndReact(LatexRendererComponent);