import * as React from "react";
import { Form, IFields, required, maxLength } from "./Form";
import { Field } from "./Field";

export const ContactUsForm: React.SFC = () => {

  const fields: IFields = {
    model_number: {
      id: "model_number",
      label: "Model",
      editor: "dropdown",
      options : ["", "0", "1"],
      optionsHelper: ["", "Neural Net", "Linear SVC"],
      validation: { rule: required }
    },
    tweet: {
      id: "name",
      label: "Tweet",
      editor: "multilinetextbox",
      validation: { rule: maxLength, args: 1000 }
    }
  };
  return (
    <Form
      action="http://184.172.233.161:31222/"
      fields={fields}
      render={() => (
        <React.Fragment>
          <Field {...fields.model_number} />
          <Field {...fields.tweet} />
        </React.Fragment>
      )}
    />
  );
};
