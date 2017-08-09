.. currentmodule:: {{ fullname }}

{% block functions %}

.. autosummary::
{% for item in functions %}
   {{ item }}
{%- endfor %}

{% endblock %}
