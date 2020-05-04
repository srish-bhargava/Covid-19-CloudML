import * as React from "react";
import { Form, IFields, required, maxLength } from "./Form";
import { Field } from "./Field";

export const ContactUsForm: React.SFC = () => {
  const fields: IFields = {
    model_number: {
      id: "model_number",
      label: "Method",
      editor: "dropdown",
      options: ["","1", "2"],
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
      action="http://localhost:4351/"
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
